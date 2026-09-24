"""
Zarr Data Lake Builder & Feature Store Adapter (Feast / Object Storage).
Persists multi-sensor, feature-engineered spatio-temporal tensors into cloud-native Zarr stores.
Uses Dask chunking optimized for ML model sequence loading (time=6, lat=256, lon=256).
"""

import os
import zarr
import xarray as xr
import dask.array as da
from typing import Dict, Any
from config.grid_config import IndiaGridConfig

class ZarrLakeBuilder:
    """Manages reading and writing multi-modal convective tensors to Zarr format."""

    def __init__(self, zarr_path: str = "data/zarr_lake/india_convective_cube.zarr"):
        self.zarr_path = zarr_path
        os.makedirs(os.path.dirname(self.zarr_path), exist_ok=True)

    def write_to_zarr(self, feature_ds: xr.Dataset, time_stamp: str) -> str:
        """Appends snapshot to Zarr store with chunking optimized for PyTorch DataLoader."""
        # Add time dimension if not present
        if "time" not in feature_ds.dims:
            feature_ds = feature_ds.expand_dims(time=[pd.to_datetime(time_stamp)])

        # Apply spatial & temporal chunking for parallel Dask reads
        chunk_dict = {
            "time": 1,
            "latitude": 256,
            "longitude": 256
        }
        
        # Chunk variables
        chunked_ds = feature_ds.chunk(chunk_dict)

        # Write to Zarr store
        if not os.path.exists(self.zarr_path):
            chunked_ds.to_zarr(self.zarr_path, mode="w")
        else:
            chunked_ds.to_zarr(self.zarr_path, mode="a", append_dim="time")

        return self.zarr_path

    def load_zarr_cube((self) -> xr.Dataset:
        """Lazy loads Zarr data cube with Dask back-end."""
        if not os.path.exists(self.zarr_path):
            raise FileNotFoundError(f"Zarr store does not exist at {self.zarr_path}")
        return xr.open_zarr(self.zarr_path)

if __name__ == "__main__":
    import pandas as pd
    from src.fusion.regridder import MultiSensorRegridder, IndiaGridConfig
    from src.ingestion.imd_dwr_ingest import IMDDWRIngestor
    from src.ingestion.mosdac_insat_ingest import MOSDACINSATIngestor
    from src.ingestion.lightning_ingest import LightningIngestor
    from src.ingestion.ncmrwf_ingest import NCMRWFIngestor
    from src.fusion.feature_extractor import ConvectiveFeatureExtractor

    grid = IndiaGridConfig()
    dwr = IMDDWRIngestor(grid).grid_to_cartesian_india(IMDDWRIngestor(grid).apply_quality_control(IMDDWRIngestor(grid).read_raw_radar_volume("")))
    insat = MOSDACINSATIngestor(grid).reproject_to_india_grid(MOSDACINSATIngestor(grid).fetch_mosdac_granule("2026-09-24T12:00:00Z"))
    lightning = LightningIngestor(grid).convert_strikes_to_flash_density(LightningIngestor(grid).generate_synthetic_strikes(500))
    ncmrwf = NCMRWFIngestor(grid).read_ncmrwf_forecast("2026-09-24T12:00:00Z")

    regridder = MultiSensorRegridder(grid)
    fused_tensor = regridder.align_and_fuse(dwr, insat, lightning, ncmrwf)
    extractor = ConvectiveFeatureExtractor()
    feature_ds = extractor.extract_features(fused_tensor)

    builder = ZarrLakeBuilder("data/zarr_lake/test_convective.zarr")
    path = builder.write_to_zarr(feature_ds, "2026-09-24T12:00:00Z")
    print(f"Zarr Data Lake successfully built/updated at: {path}")
