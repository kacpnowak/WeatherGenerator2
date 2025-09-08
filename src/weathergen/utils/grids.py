from datetime import timedelta

import numpy as np
from astropy_healpix import HEALPix

from weathergen.datasets.data_reader_base import DTRange


def regular_grid_coords(specs: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Create coordiates of regular grid

    Args:
        specs (dict): dictionary with grid specification

    Returns:
        np.ndarray, np.ndarray: latitude, longitude
    """
    lat_res = 1
    lon_res = 1
    if "res" in specs:
        lat_res = int(specs["res"])
        lon_res = lat_res
    elif "lat_res" in specs and "lon_res" in specs:
        lat_res = int(specs["lat_res"])
        lon_res = int(specs["lon_res"])
    else:
        raise AttributeError("Specification must contain res or lat_res and lon_res")

    lat_min = specs["lat_min"] if "lat_min" in specs else 0
    lat_max = specs["lat_max"] if "lat_max" in specs else 180

    lon_min = specs["lon_min"] if "lon_min" in specs else 0
    lon_max = specs["lon_max"] if "lon_max" in specs else 360

    lat = np.arange(lat_min, lat_max, lat_res)
    lon = np.arange(lon_min, lon_max, lon_res)

    nx = len(lon)
    ny = len(lat)

    lats = np.zeros((nx, ny))
    lons = lats.copy()

    for i in range(nx):
        lats[i, :] = lat

    for i in range(ny):
        lons[:, i] = lon

    return lats.flatten(), lons.flatten()


def healpix_coords(specs: dict) -> tuple[np.ndarray, np.ndarray]:
    """
    Create coordinates of healpix mesh

    Args:
    specs (dict): dictionary with grid specification

    Returns:
        np.ndarray, np.ndarray: latitude, longitude
    """
    level = 1
    if "level" in specs:
        level = int(specs["level"])
    else:
        raise AttributeError("Specification must contain level for healpix mesh")

    hp = HEALPix(nside=2**level, order="nested", frame=None)
    # Pixel indices
    pix = np.arange(hp.npix)

    # Get spherical coordinates (lon, lat)
    lon, lat = hp.healpix_to_lonlat(pix)
    return np.degrees(lat) + 90.0, np.degrees(lon)


def temporal_coords(range: DTRange, specs: dict) -> np.ndarray:
    """
    Creates a temporal coordinates based on specs

    Args:
        handler (DTRange): range within the timesteps should be generated
        specs (dict): dictionary with grid specification

    Returns:
        np.ndarray: datetimes
    """
    if "time_res" in specs:
        res = int(specs["time_res"])
        start = range.start
        end = range.end
        return np.arange(start, end, timedelta(hours=res)).astype(np.datetime64)
    else:
        raise AttributeError("Temporal resolution must be specified with time_res")
