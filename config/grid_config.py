"""
Grid Configuration and Spatial Utilities for India Domain Nowcasting.
Provides standard spatial grid coordinates (68-98°E, 6-38°N) and configuration loader.
"""

import os
import yaml
import numpy as np
from typing import Dict, Any, Tuple

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "settings.yaml")

def load_settings(config_path: str = CONFIG_PATH) -> Dict[str, Any]:
    """Loads system settings from YAML file."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found at {config_path}")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

class IndiaGridConfig:
    """Standard spatial grid definition over the Indian subcontinent."""
    
    def __init__(self, settings: Dict[str, Any] = None):
        if settings is None:
            settings = load_settings()
        
        spatial_cfg = settings["spatial_grid"]
        self.min_lon = float(spatial_cfg["min_lon"])
        self.max_lon = float(spatial_cfg["max_lon"])
        self.min_lat = float(spatial_cfg["min_lat"])
        self.max_lat = float(spatial_cfg["max_lat"])
        self.resolution = float(spatial_cfg["resolution_deg"])
        self.projection = spatial_cfg["projection"]
        
        # Grid dimensions
        self.lons = np.arange(self.min_lon, self.max_lon + self.resolution / 2, self.resolution)
        self.lats = np.arange(self.min_lat, self.max_lat + self.resolution / 2, self.resolution)
        self.shape = (len(self.lats), len(self.lons))
        
    def get_mesh(self) -> Tuple[np.ndarray, np.ndarray]:
        """Returns 2D Meshgrid of (Longitude, Latitude)."""
        return np.meshgrid(self.lons, self.lats)

    def get_bounds(self) -> Tuple[float, float, float, float]:
        """Returns (min_lon, min_lat, max_lon, max_lat)."""
        return self.min_lon, self.min_lat, self.max_lon, self.max_lat

    def __repr__(self) -> str:
        return (f"<IndiaGridConfig: bounds=({self.min_lon}°E - {self.max_lon}°E, "
                f"{self.min_lat}°N - {self.max_lat}°N), res={self.resolution}°, shape={self.shape}>")

if __name__ == "__main__":
    grid = IndiaGridConfig()
    print(grid)
