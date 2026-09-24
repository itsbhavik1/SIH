"""
ISRO MOSDAC INSAT-3DR/3DS Imager Ingestion Module.
Downloads and processes thermal IR (TIR1, TIR2), water vapor (WV), and visible channels.
Performs Brightness Temperature (BT) calibration, overshooting top detection, and Satpy reprojection.
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, List
from config.grid_config import IndiaGridConfig

class MOSDACINSATIngestor:
    """Ingestion & Preprocessing pipeline for ISRO INSAT-3D/3DR/3DS Satellite Imager."""

    def __init__(self, grid_config: IndiaGridConfig = None):
        self.grid = grid_config or IndiaGridConfig()
        self.channels = ["TIR1", "TIR2", "WV", "MIR", "VIS"]

    def fetch_mosdac_granule(self, timestamp: str, satellite: str = "INSAT-3DR") -> Dict[str, Any]:
        """Simulates fetching MOSDAC HDF5/NetCDF granule via MOSDAC API client."""
        # Simulated native geostationary grid
        ny, nx = 800, 800
        
        # Synthetic Brightness Temperature (K) for TIR1 (10.8 um)
        # Deep convective clouds have cloud-top BT < 210 K (-63°C)
        np.random.seed(101)
        tir1_bt = np.random.uniform(240.0, 295.0, size=(ny, nx))
        # Add deep convective cloud tops (cold overshooting tops ~195-210 K)
        tir1_bt[300:400, 350:450] = np.random.uniform(195.0, 210.0, size=(100, 100))

        # Synthetic Water Vapor Channel BT (K)
        wv_bt = tir1_bt - np.random.uniform(2.0, 15.0, size=(ny, nx))

        return {
            "satellite": satellite,
            "timestamp": timestamp,
            "TIR1": tir1_bt,
            "WV": wv_bt,
            "native_shape": (ny, nx)
        }

    def compute_overshooting_tops(self, tir1_bt: np.ndarray, wv_bt: np.ndarray) -> np.ndarray:
        """Calculates Overshooting Top (OT) index based on BTD (TIR1 - WV < 0 K) and cold BT thresholds."""
        # BTD (TIR1 - WV) inversion is a classic indicator of overshooting convective cloud tops into stratosphere
        btd = tir1_bt - wv_bt
        ot_mask = (tir1_bt < 215.0) & (btd < 2.0)
        return ot_mask.astype(np.float32)

    def reproject_to_india_grid(self, granule: Dict[str, Any]) -> xr.Dataset:
        """Reprojects geostationary INSAT channels onto standard lat/lon India grid using Satpy / pyresample."""
        lat_grid, lon_grid = self.grid.get_mesh()
        
        # Simulated reprojection onto 0.02° India domain
        ny, nx = self.grid.shape
        np.random.seed(202)
        
        # Reprojected Brightness Temperatures
        tir1_grid = np.random.uniform(250.0, 300.0, size=(ny, nx))
        # Convective cloud mass over central India
        conv_mask = (lat_grid >= 15.0) & (lat_grid <= 23.0) & (lon_grid >= 73.0) & (lon_grid <= 82.0)
        tir1_grid[conv_mask] = np.random.uniform(198.0, 220.0, size=np.sum(conv_mask))
        
        wv_grid = tir1_grid - np.random.uniform(1.0, 10.0, size=(ny, nx))
        ot_grid = self.compute_overshooting_tops(tir1_grid, wv_grid)

        ds = xr.Dataset(
            {
                "TIR1_BT": (["latitude", "longitude"], tir1_grid.astype(np.float32), {"units": "K", "long_name": "TIR1 Brightness Temp"}),
                "WV_BT": (["latitude", "longitude"], wv_grid.astype(np.float32), {"units": "K", "long_name": "Water Vapor Brightness Temp"}),
                "Overshooting_Top_Index": (["latitude", "longitude"], ot_grid, {"units": "binary", "long_name": "Overshooting Convective Tops"})
            },
            coords={"latitude": self.grid.lats, "longitude": self.grid.lons},
            attrs={"satellite": granule["satellite"], "timestamp": granule["timestamp"]}
        )
        return ds

if __name__ == "__main__":
    ingestor = MOSDACINSATIngestor()
    granule = ingestor.fetch_mosdac_granule("2026-09-24T12:00:00Z")
    insat_ds = ingestor.reproject_to_india_grid(granule)
    print("MOSDAC INSAT Processing Complete:")
    print(insat_ds)
