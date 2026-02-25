"""
Path configuration module that reads directory paths from dirs.txt
"""
import os
from pathlib import Path
from typing import Dict, Optional

# Cache for parsed paths
_PATHS_CACHE: Optional[Dict[str, str]] = None


def _get_dirs_file_path() -> Path:
    """Get the path to the dirs.txt file."""
    current_dir = Path(__file__).parent
    return current_dir / "dirs.txt"


def load_paths(force_reload: bool = False) -> Dict[str, str]:
    """
    Load directory paths from dirs.txt file.

    Args:
        force_reload: If True, reload from file even if cached

    Returns:
        Dictionary mapping path keys to their values
    """
    global _PATHS_CACHE

    if _PATHS_CACHE is not None and not force_reload:
        return _PATHS_CACHE

    paths = {}
    dirs_file = _get_dirs_file_path()

    if not dirs_file.exists():
        raise FileNotFoundError(
            f"dirs.txt not found at {dirs_file}. "
            "Please create this file with your directory configuration."
        )

    with open(dirs_file, 'r') as f:
        for line in f:
            line = line.strip()
            # Skip empty lines and comments
            if not line or line.startswith('#'):
                continue

            # Parse KEY=VALUE format
            if '=' in line:
                key, value = line.split('=', 1)
                paths[key.strip()] = value.strip()

    _PATHS_CACHE = paths
    return paths


def get_path(key: str, default: Optional[str] = None) -> str:
    """
    Get a directory path by its key.

    Args:
        key: The key for the path (e.g., 'DATA_HLS_COMPOSITES')
        default: Default value if key not found (if None, raises KeyError)

    Returns:
        The path string

    Raises:
        KeyError: If key not found and no default provided
    """
    paths = load_paths()

    if key not in paths:
        if default is not None:
            return default
        raise KeyError(
            f"Path key '{key}' not found in dirs.txt. "
            f"Available keys: {list(paths.keys())}"
        )

    return paths[key]


def get_mean_stds_dir() -> str:
    return get_path("MEAN_STDS_DIR")

def get_data_paths_dir() -> str:
    return get_path("DATA_PATHS_DIR")

def get_data_hls_composites() -> str:
    """Get the HLS composites data directory."""
    return get_path('DATA_HLS_COMPOSITES')

def get_data_lsp_ancillary() -> str:
    """Get the LSP ancillary data directory."""
    return get_path('DATA_LSP_ANCILLARY')

def get_data_geojson() -> str:
    """Get the geotiff extents geojson file path."""
    return get_path('DATA_GEOJSON')

def get_checkpoint_root() -> str:
    """Get the checkpoint root directory."""
    return get_path('CHECKPOINT_ROOT')

def get_pixels_cache_dir() -> str:
    """Get the checkpoint root directory."""
    return get_path('PIXELS_CACHE_DIR')


def get_model_weights(model_size: str) -> str:
    """
    Get the model weights file path.

    Args:
        model_size: '300m' or '600m'

    Returns:
        Path to model weights file
    """
    key = f'MODEL_WEIGHTS_{model_size.upper()}'
    return get_path(key)


if __name__ == '__main__':
    # Test loading
    paths = load_paths()
    print("Loaded paths from dirs.txt:")
    for key, value in paths.items():
        print(f"  {key} = {value}")
