"""
Grid Configuration and Spatial Utilities for India Domain Nowcasting.
Provides standard spatial grid coordinates (68-98°E, 6-38°N), resolution definitions,
coordinate conversion utilities, and area geometry specifications.
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, Tuple, Optional

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "settings.yaml")

def load_settings(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Loads system settings from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class IndiaGridConfig:
    """
    Standard spatial grid definition over the Indian subcontinent domain.
    Domain Bounding Box: 68.0°E to 98.0°E, 6.0°N to 38.0°N.
    Target Spatial Resolution: 0.02° (~2.2 km per cell at equator).
    """

    def __init__(self, settings: Optional[Dict[str, Any]] = None):
        if settings is None:
            settings = load_settings()

        spatial_cfg = settings["spatial_grid"]
        self.min_lon = float(spatial_cfg["min_lon"])
        self.max_lon = float(spatial_cfg["max_lon"])
        self.min_lat = float(spatial_cfg["min_lat"])
        self.max_lat = float(spatial_cfg["max_lat"])
        self.resolution = float(spatial_cfg["resolution_deg"])
        self.projection = spatial_cfg.get("projection", "EPSG:4326")

        # Derive 1D coordinate vectors
        self.lons = np.arange(self.min_lon, self.max_lon + self.resolution / 2.0, self.resolution)
        self.lats = np.arange(self.min_lat, self.max_lat + self.resolution / 2.0, self.resolution)
        self.shape = (len(self.lats), len(self.lons))

        # Spatial Extents
        self.n_lats, self.n_lons = self.shape

    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns 2D Meshgrid of (Longitude, Latitude) coordinates."""
        return np.meshgrid(self.lons, self.lats)

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Returns (min_lon, min_lat, max_lon, max_lat)."""
        return self.min_lon, self.min_lat, self.max_lon, self.max_lat

    def latlon_to_index(self, lat: float, lon: float) -> Tuple[int, int]:
        """Converts geographic latitude/longitude to grid cell indices (lat_idx, lon_idx)."""
        lat_idx = int(np.clip(round((lat - self.min_lat) / self.resolution), 0, self.n_lats - 1))
        lon_idx = int(np.clip(round((lon - self.min_lon) / self.resolution), 0, self.n_lons - 1))
        return lat_idx, lon_idx

    def index_to_latlon(self, lat_idx: int, lon_idx: int) -> Tuple[float, float]:
        """Converts grid cell indices back to geographic latitude/longitude."""
        lat = self.min_lat + lat_idx * self.resolution
        lon = self.min_lon + lon_idx * self.resolution
        return float(lat), float(lon)

    def get_pyresample_area_def(self) -> Dict[str, Any]:
        """Returns geometry parameters required for pyresample / Satpy spatial reprojection."""
        return {
            "area_id": "india_convective_domain",
            "description": "Standard India Convective Nowcasting Domain 0.02 deg",
            "proj_id": "EPSG:4326",
            "projection": {"proj": "latlong", "datum": "WGS84"},
            "width": self.n_lons,
            "height": self.n_lats,
            "area_extent": (self.min_lon, self.min_lat, self.max_lon, self.max_lat)
        }

    def __repr__(self) -> str:
        return (
            f"<IndiaGridConfig: bounds=({self.min_lon}°E - {self.max_lon}°E, "
            f"{self.min_lat}°N - {self.max_lat}°N), resolution={self.resolution}°, shape={self.shape}>"
        )

if __name__ == "__main__":
    grid = IndiaGridConfig()
    print(grid)
    print("PyResample Area Def:", grid.get_pyresample_area_def())
