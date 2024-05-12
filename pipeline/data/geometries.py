from functools import reduce
from pathlib import Path
from typing import Union

import geopandas as gpd
import numpy as np
import rasterio
from shapely import Polygon


def get_bbox_size(tif_path: Path, _bbox) -> tuple:
    with rasterio.open(tif_path) as dataset:
        window = rasterio.windows.from_bounds(*_bbox, transform=dataset.transform)
    height, width = int(np.round(window.height)), int(np.round(window.width))
    return height, width


def fill_geometry(df_row: gpd.GeoSeries, size_lim):
    """A function to fill holes below an area threshold in a polygon"""
    new_geom = None
    rings = [i for i in df_row["geometry"].interiors]  # List all interior rings
    if len(rings) > 0:  # If there are any rings
        to_fill = [Polygon(ring) for ring in rings if Polygon(ring).area < size_lim]  # List the ones to fill
        if len(to_fill) > 0:  # If there are any to fill
            new_geom = reduce(
                lambda geom1, geom2:
                geom1.union(geom2),
                [df_row["geometry"]] + to_fill
            )  # Union the original geometry with all holes
    if new_geom:
        return new_geom
    else:
        return df_row["geometry"]


def load_lake_geometry_from_file(lake_geometry_fp: Union[str, Path]):
    lake_gdf = gpd.read_file(lake_geometry_fp)
    return lake_gdf
