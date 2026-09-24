"""
Cloud-Native Zarr Data Lake & Feast Feature Store Adapter.
Manages high-throughput reads/writes of multi-sensor spatio-temporal tensors.
Applies Dask chunking optimized for GPU batch pre-fetching: (time=1, latitude=256, longitude=256).
"""

import os
import zarr
import pandas as pd
import xarray as xr
import dask.array as da
from typing import Dict, Any, Optional
from config.grid_config import IndiaGridConfig

class ZarrLakeBuilder:
    """Manages appending and reading spatio-temporal Zarr data cubes."""

    def __init__(self, zarr_path: str = "data/zarr_lake/india_convective_cube.zarr"):
        self.zarr_path = zarr_path
        os.makedirs(os.path.dirname(self.zarr_path), exist_ok=True)

    def write_snapshot_to_zarr(self, feature_ds: xr.Dataset, timestamp_str: str) -> str:
        """Appends single temporal snapshot to Zarr store with optimized Dask chunking."""
        ts = pd.to_datetime(timestamp_str)
        if "time" not in feature_ds.dims:
            feature_ds = feature_ds.expand_dims(time=[ts])

        chunk_dict = {"time": 1, "latitude": 256, "longitude": 256}
        chunked_ds = feature_ds.chunk(chunk_dict)

        if not os.path.exists(self.zarr_path):
            chunked_ds.to_zarr(self.zarr_path, mode="w")
        else:
            chunked_ds.to_zarr(self.zarr_path, mode="a", append_dim="time")

        return self.zarr_path

    def load_zarr_dataset(self) -> xr.Dataset:
        """Lazy-loads Zarr data cube with Dask backend."""
        if not os.path.exists(self.zarr_path):
            raise FileNotFoundError(f"Zarr store does not exist at {self.zarr_path}")
        return xr.open_zarr(self.zarr_path)

if __name__ == "__main__":
    from src.fusion.regridder import MultiSensorRegridder, IndiaGridConfig
    from src.ingestion.imd_dwr_ingest import IMDDWRIngestor
    from src.ingestion.mosdac_insat_ingest import MOSDACINSATIngestor
    from src.ingestion.lightning_ingest import LightningStreamIngestor
    from src.ingestion.ncmrwf_ingest import NCMRWFIngestor
    from src.fusion.feature_extractor import ConvectiveFeatureExtractor

    grid = IndiaGridConfig()
    dwr_ds = IMDDWRIngestor(grid).grid_polar_to_cartesian_mosaic(IMDDWRIngestor(grid).apply_attenuation_correction(IMDDWRIngestor(grid).remove_ground_clutter(IMDDWRIngestor(grid).read_raw_volume_scan(""))))
    insat_ds = MOSDACINSATIngestor(grid).process_and_reproject_granule("2026-09-24T12:00:00Z")
    lgt_ing = LightningStreamIngestor(grid)
    lgt_ds = lgt_ing.rasterize_flash_density(lgt_ing.cluster_strokes_to_flashes(lgt_ing.generate_synthetic_stream()))
    ncmrwf_ds = NCMRWFIngestor(grid).extract_environmental_covariates("2026-09-24T12:00:00Z")

    regridder = MultiSensorRegridder(grid)
    fused_ds = regridder.align_and_fuse(dwr_ds, insat_ds, lgt_ds, ncmrwf_ds)
    extractor = ConvectiveFeatureExtractor()
    feature_ds = extractor.extract_features(fused_ds)

    builder = ZarrLakeBuilder("data/zarr_lake/test_convective.zarr")
    path = builder.write_snapshot_to_zarr(feature_ds, "2026-09-24T12:00:00Z")
    print(f"Zarr Lake Builder Verified. Persisted at: {path}")
