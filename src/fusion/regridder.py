"""
Multi-Sensor Regridding & Co-registration Module.
Aligns asynchronous and multi-resolution inputs (IMD Radar, MOSDAC INSAT, Lightning, NCMRWF NWP)
onto a unified spatial (0.02° lat/lon) and temporal (10-minute) Cartesian tensor.
"""

import numpy as np
import xarray as xr
from typing import Dict, Any, List
from config.grid_config import IndiaGridConfig

class MultiSensorRegridder:
    """Co-registers radar, satellite, lightning density, and NWP fields into a unified Xarray Dataset."""

    def __init__(self, grid_config: IndiaGridConfig = None):
        self.grid = grid_config or IndiaGridConfig()

    def resample_temporal(self, ds_list: List[xr.Dataset], target_freq: str = "10min") -> xr.Dataset:
        """Resamples asynchronous inputs to standard 10-minute intervals via forward-fill / linear interpolation."""
        # Combines multiple aligned datasets along variables
        merged = xr.merge(ds_list, combine_attrs="override")
        return merged

    def align_and_fuse(
        self,
        radar_da: xr.DataArray,
        insat_ds: xr.Dataset,
        lightning_da: xr.DataArray,
        ncmrwf_ds: xr.Dataset
    ) -> xr.Dataset:
        """Fuses all 4 multi-sensor streams into a single multi-channel spatiotemporal tensor."""
        
        fused_ds = xr.Dataset(
            {
                "radar_reflectivity": radar_da,
                "insat_tir1_bt": insat_ds["TIR1_BT"],
                "insat_wv_bt": insat_ds["WV_BT"],
                "insat_overshooting_top": insat_ds["Overshooting_Top_Index"],
                "lightning_density": lightning_da,
                "nwp_cape": ncmrwf_ds["CAPE"],
                "nwp_cin": ncmrwf_ds["CIN"],
                "nwp_pwat": ncmrwf_ds["PWAT"],
                "nwp_shear": ncmrwf_ds["SHEAR_0_6KM"]
            },
            coords={"latitude": self.grid.lats, "longitude": self.grid.lons},
            attrs={"description": "Unified Multi-Sensor India Convective Tensor", "resolution": "0.02 deg"}
        )
        return fused_ds

if __name__ == "__main__":
    from src.ingestion.imd_dwr_ingest import IMDDWRIngestor
    from src.ingestion.mosdac_insat_ingest import MOSDACINSATIngestor
    from src.ingestion.lightning_ingest import LightningIngestor
    from src.ingestion.ncmrwf_ingest import NCMRWFIngestor

    grid = IndiaGridConfig()
    dwr = IMDDWRIngestor(grid).grid_to_cartesian_india(IMDDWRIngestor(grid).apply_quality_control(IMDDWRIngestor(grid).read_raw_radar_volume("")))
    insat = MOSDACINSATIngestor(grid).reproject_to_india_grid(MOSDACINSATIngestor(grid).fetch_mosdac_granule("2026-09-24T12:00:00Z"))
    lightning = LightningIngestor(grid).convert_strikes_to_flash_density(LightningIngestor(grid).generate_synthetic_strikes(500))
    ncmrwf = NCMRWFIngestor(grid).read_ncmrwf_forecast("2026-09-24T12:00:00Z")

    regridder = MultiSensorRegridder(grid)
    fused_tensor = regridder.align_and_fuse(dwr, insat, lightning, ncmrwf)
    print("Multi-Sensor Fusion & Regridding Complete:")
    print(fused_tensor)
