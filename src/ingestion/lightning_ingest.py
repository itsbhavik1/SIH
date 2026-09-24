"""
Lightning Data Ingestion & Flash Density Gridding Module.
Handles Indian Lightning Location Network (ILLN - IITM Pune) & public fallback (GPM-LIS / WWLLN).
Converts point-strike strike events into temporal flash-density rasters (strokes / km² / 10-min).
"""

import numpy as np
import pandas as pd
import xarray as xr
from typing import Dict, Any, List
from config.grid_config import IndiaGridConfig

class LightningIngestor:
    """Ingestor for ground-based (ILLN, WWLLN) and space-borne (GPM-LIS) lightning strikes."""

    def __init__(self, grid_config: IndiaGridConfig = None):
        self.grid = grid_config or IndiaGridConfig()

    def generate_synthetic_strikes(self, count: int = 500) -> pd.DataFrame:
        """Simulates incoming Kafka stream of lightning strike point events over India."""
        np.random.seed(303)
        # Cluster lightning strikes around active convective zone (Central/West India)
        lats = np.random.normal(loc=19.5, scale=1.5, size=count)
        lons = np.random.normal(loc=74.5, scale=1.5, size=count)
        
        # Clip to valid bounding box
        lats = np.clip(lats, self.grid.min_lat, self.grid.max_lat)
        lons = np.clip(lons, self.grid.min_lon, self.grid.max_lon)
        
        peak_current_ka = np.random.exponential(scale=25.0, size=count) * np.random.choice([-1, 1], size=count)
        
        df = pd.DataFrame({
            "timestamp": pd.date_range("2026-09-24 12:00:00", periods=count, freq="1s"),
            "latitude": lats,
            "longitude": lons,
            "peak_current_ka": peak_current_ka,
            "network": "ILLN_IITM"
        })
        return df

    def convert_strikes_to_flash_density(self, strike_df: pd.DataFrame, time_window_min: int = 10) -> xr.DataArray:
        """Grids point lightning events onto 2D spatial grid producing flash density (flashes/km²/window)."""
        lat_bins = np.append(self.grid.lats - self.grid.resolution / 2, self.grid.lats[-1] + self.grid.resolution / 2)
        lon_bins = np.append(self.grid.lons - self.grid.resolution / 2, self.grid.lons[-1] + self.grid.resolution / 2)
        
        # 2D Histogram of strike counts per grid cell
        counts, _, _ = np.histogram2d(
            strike_df["latitude"].values,
            strike_df["longitude"].values,
            bins=[lat_bins, lon_bins]
        )
        
        # Grid cell area in km² (~2.22 km x 2.22 km at 0.02 deg ~ 4.9 km²)
        cell_area_km2 = (self.grid.resolution * 111.0) ** 2
        flash_density = counts / cell_area_km2 # flashes per km²

        da = xr.DataArray(
            flash_density.astype(np.float32),
            coords=[("latitude", self.grid.lats), ("longitude", self.grid.lons)],
            name="lightning_flash_density",
            attrs={
                "units": "flashes / km2 / 10min",
                "long_name": "Cloud-to-Ground Lightning Flash Density",
                "source_network": "ILLN/GPM-LIS"
            }
        )
        return da

if __name__ == "__main__":
    ingestor = LightningIngestor()
    strikes = ingestor.generate_synthetic_strikes(count=1200)
    density_da = ingestor.convert_strikes_to_flash_density(strikes)
    print("Lightning Flash Density Processing Complete:")
    print(density_da)
    print(f"Max Flash Density: {density_da.max().item():.4f} flashes/km²")
