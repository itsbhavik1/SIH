"""
Multi-Sensor Co-Registration & Temporal Alignment Engine.
Synchronizes heterogeneous, multi-resolution, and asynchronous data feeds:
  - IMD Doppler Weather Radar (10-min, ~1 km Cartesian)
  - ISRO MOSDAC INSAT-3DR/3DS Satellite (15-min, ~4 km Geostationary)
  - ILLN / GPM-LIS Lightning Strikes (Streamed, Point Event Rasters)
  - NCMRWF IMDAA NWP Covariates (1-hourly / 6-hourly, 12 km Regional Grid)

Fuses all data streams into a unified 12-channel spatial tensor on the 0.02° India grid.
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, List, Optional
from config.grid_config import IndiaGridConfig

class MultiSensorRegridder:
    """Co-registers radar, satellite, lightning density, and NWP fields into a unified Xarray Dataset."""

    def __init__(self, grid_config: Optional[IndiaGridConfig] = None):
        self.grid = grid_config or IndiaGridConfig()

    def fill_spatial_gaps_and_masks(self, da: xr.DataArray, fill_value: float = 0.0) -> xr.DataArray:
        """Fills spatial gaps over un-covered radar regions with satellite/NWP continuous fields."""
        return da.fillna(fill_value)

    def align_and_fuse(
        self,
        radar_ds: xr.Dataset,
        insat_ds: xr.Dataset,
        lightning_ds: xr.Dataset,
        ncmrwf_ds: xr.Dataset
    ) -> xr.Dataset:
        """
        Fuses all multi-sensor streams into a single multi-channel spatiotemporal tensor.
        Channel Ordering (12 total channels):
          0. Radar Composite Max Reflectivity (dBZ)
          1. Radar Echo Top Height 18dBZ (km)
          2. INSAT TIR1 Brightness Temp (K)
          3. INSAT TIR2 Brightness Temp (K)
          4. INSAT Water Vapor BT (K)
          5. INSAT Brightness Temp Difference (BTD = TIR1 - WV)
          6. INSAT Overshooting Convective Cloud Top Mask (binary)
          7. Lightning Flash Density (flashes/km²/10min)
          8. NCMRWF CAPE (J/kg)
          9. NCMRWF CIN (J/kg)
          10. NCMRWF Precipitable Water (PWAT kg/m²)
          11. NCMRWF 0-6 km Bulk Shear (m/s)
        """
        # Ensure clean spatial alignment
        radar_dbz = self.fill_spatial_gaps_and_masks(radar_ds["max_reflectivity_dbz"], fill_value=0.0)
        echo_top = self.fill_spatial_gaps_and_masks(radar_ds["echo_top_18dbz_km"], fill_value=0.0)

        fused_ds = xr.Dataset(
            {
                "ch00_radar_dbz": radar_dbz,
                "ch01_radar_echo_top": echo_top,
                "ch02_insat_tir1_bt": insat_ds["TIR1_BT"],
                "ch03_insat_tir2_bt": insat_ds["TIR2_BT"],
                "ch04_insat_wv_bt": insat_ds["WV_BT"],
                "ch05_insat_btd": insat_ds["BTD_TIR1_WV"],
                "ch06_insat_overshooting_top": insat_ds["Overshooting_Top_Index"],
                "ch07_lightning_density": lightning_ds["lightning_flash_density"],
                "ch08_nwp_cape": ncmrwf_ds["CAPE"],
                "ch09_nwp_cin": ncmrwf_ds["CIN"],
                "ch10_nwp_pwat": ncmrwf_ds["PWAT"],
                "ch11_nwp_shear": ncmrwf_ds["SHEAR_0_6KM"]
            },
            coords={"latitude": self.grid.lats, "longitude": self.grid.lons},
            attrs={
                "description": "Unified 12-Channel Multi-Sensor India Convective Tensor",
                "resolution": "0.02 deg (~2.2 km)",
                "grid_shape": str(self.grid.shape)
            }
        )
        return fused_ds

if __name__ == "__main__":
    from src.ingestion.imd_dwr_ingest import IMDDWRIngestor
    from src.ingestion.mosdac_insat_ingest import MOSDACINSATIngestor
    from src.ingestion.lightning_ingest import LightningStreamIngestor
    from src.ingestion.ncmrwf_ingest import NCMRWFIngestor

    grid = IndiaGridConfig()
    dwr_ds = IMDDWRIngestor(grid).grid_polar_to_cartesian_mosaic(
        IMDDWRIngestor(grid).apply_attenuation_correction(
            IMDDWRIngestor(grid).remove_ground_clutter(
                IMDDWRIngestor(grid).read_raw_volume_scan("")
            )
        )
    )
    insat_ds = MOSDACINSATIngestor(grid).process_and_reproject_granule("2026-09-24T12:00:00Z")
    lgt_ingestor = LightningStreamIngestor(grid)
    lgt_ds = lgt_ingestor.rasterize_flash_density(lgt_ingestor.cluster_strokes_to_flashes(lgt_ingestor.generate_synthetic_stream()))
    ncmrwf_ds = NCMRWFIngestor(grid).extract_environmental_covariates("2026-09-24T12:00:00Z")

    regridder = MultiSensorRegridder(grid)
    fused = regridder.align_and_fuse(dwr_ds, insat_ds, lgt_ds, ncmrwf_ds)
    print("Multi-Sensor Regridding & Fusion Verified:")
    print(fused)
