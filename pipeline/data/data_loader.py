"""
Module for loading and processing DEM (Digital Elevation Model) data for lakes.

The `DataLoader` class takes in lake ID, buffer size, output directory, and optional
date and lake geometry parameters. It loads DEM data from tiles and strips,
applies filtering based on lake geometry and variance threshold, and returns a
processed `xarray.DataArray` object along with a `pandas.DataFrame` containing
statistics.
"""
from typing import Union, Optional, Any
import numpy as np
import rasterio
import shapely
import xarray as xr
from affine import Affine
from geopandas import GeoDataFrame
from loguru import logger
from pandas import DataFrame
from rasterio.features import rasterize
from shapely import Polygon, MultiPolygon
from tqdm import tqdm
from xarray import DataArray

from data.geometries import get_bbox_size

from utils import execution_utils
from utils.constants import *
from data.retriever import get_dem_tif_url_from_fileurl, get_lake_geometries_by_id, get_strips_by_lake_id, \
    get_tiles_by_lake_id


class DataLoader:
    def __init__(self, lake_id: Union[int, str], buffer: Union[float, int], output_dir: Path,
                 date: Optional[pd.Timestamp] = None,
                 lake_geometry: Optional = None, variance_reject: bool = False):
        self.lake_id = lake_id
        self.buffer = buffer
        self.output_dir = output_dir
        self.date = date
        self.lake_geometry = lake_geometry
        self.variance_reject = variance_reject

    def load_dem_array(self) -> tuple[DataArray, DataFrame]:
        date_str = self.date.strftime("%Y-%m-%d") if self.date else "all"
        lake_query_hull, tile_query_gdf = get_tiles_by_lake_id(
            lake_id=self.lake_id,
            buffer=self.buffer,
            date_str=date_str,
            lake_geometry=self.lake_geometry,
            output_dir=self.output_dir
        )

        _, strip_query_gdf = get_strips_by_lake_id(
            lake_id=self.lake_id,
            buffer=self.buffer,
            date_str=date_str,
            output_dir=self.output_dir,
            lake_geometry=self.lake_geometry,
        )
        if isinstance(self.lake_geometry, GeoDataFrame):
            lake_polygon = self.lake_geometry.geometry
            if len(lake_polygon) > 1:
                raise NotImplementedError("Only one geometry so far")
            lake_polygon = lake_polygon.squeeze()

        else:
            lake_geometries = get_lake_geometries_by_id(
                self.lake_id, output_dir=self.output_dir, date_str=date_str
            )

            if self.date in lake_geometries["date"].unique():
                # if no date is specified get lake geometry at its larges extent
                lake_polygon = lake_geometries[lake_geometries["date"] == self.date].squeeze().geometry
            else:
                lake_polygon = lake_geometries.iloc[lake_geometries.area.argmax()].geometry

        bbox = lake_query_hull.buffer(self.buffer).bounds

        if isinstance(lake_polygon, MultiPolygon):
            # shapely.MultiPolygon([poly.exterior for poly in lake_polygon.geoms])
            shapely.MultiPolygon([poly for poly in lake_polygon.geoms])
        elif isinstance(lake_polygon, Polygon):
            lake_polygon = Polygon(list(lake_polygon.exterior.coords))

        # Tile & Strip URLs / Paths to use
        strip_paths = strip_query_gdf.fileurl.apply(get_dem_tif_url_from_fileurl)
        tile_paths = tile_query_gdf.fileurl.apply(get_dem_tif_url_from_fileurl)
        dem_paths = pd.concat([strip_paths, tile_paths], ignore_index=True)

        n_dems = len(dem_paths)
        logger.info(f"Found {n_dems} strips and tiles containing #{self.lake_id}")
        bbox_size = get_bbox_size(dem_paths.iloc[-1], bbox)  # selects last tif to set the reading window size

        # Datastacks to fill:
        stats = pd.DataFrame(columns=["n_nan", "mean", "var", "date"])
        dem_stack = np.empty(shape=[n_dems, *bbox_size])
        mask_stack = np.empty(shape=[n_dems, *bbox_size])
        exterior_stack = np.empty(shape=[n_dems, *bbox_size])

        for i, dem_file in tqdm(enumerate(dem_paths), desc="Loading DEMs", total=n_dems):
            if "mosaic" in str(dem_file):
                date = REF_DATE
            else:
                date = execution_utils.extract_timestamp_from_stem(dem_file.stem)
            with rasterio.open(dem_file) as dataset:
                stats.at[i, "crs_epsg"] = dataset.crs.to_string()
                window = rasterio.windows.from_bounds(*bbox, transform=dataset.transform)

                # geometry filtering: remove any interior polygons (holes)
                # by using only the exterior coordinates
                # geometry = Polygon(shape(lake_feature.geometry).exterior)
                t = dataset.transform
                upper_left = tuple(
                    idx - self.buffer // 2 for idx in dataset.index(*lake_polygon.bounds[0:2]))  # possible pitfall
                lower_right = tuple(idx - self.buffer // 2 for idx in dataset.index(*lake_polygon.bounds[2:4]))
                shifted_affine = Affine(t.a, t.b, t.c + upper_left[1] * t.a, t.d, t.e, t.f + lower_right[0] * t.e)
                subset = dataset.read(window=window).squeeze()

                if subset.shape < dem_stack[i].shape:
                    subset = np.pad(subset, (
                        (0, dem_stack[i].shape[0] - subset.shape[0]),
                        (0, dem_stack[i].shape[1] - subset.shape[1])),
                                    constant_values=np.nan
                                    )
                mask = rasterize(
                    [(lake_polygon, 0)],
                    out_shape=subset.shape,
                    transform=shifted_affine,
                    fill=1,
                    all_touched=True,
                    dtype=np.uint8
                )

                mask_stack[i] = np.where(~mask.astype(bool), subset, np.nan).squeeze()
                exterior_stack[i] = np.where(mask.astype(bool), subset, np.nan).squeeze()
                dem_stack[i] = subset

            stats.at[i, "mean"] = np.mean(subset)
            stats.at[i, "n_nan"] = np.isnan(subset).sum() / subset.size
            stats.at[i, "var"] = np.var(subset)
            stats.at[i, "date"] = date

        dem_array = xr.DataArray(
            data=np.concatenate((
                dem_stack[np.newaxis, :],
                mask_stack[np.newaxis, :],
                exterior_stack[np.newaxis, :])
            ),
            dims=('masked', 'date', 'y', 'x'),
            coords={'date': stats["date"].values, 'masked': ["full", "masked", "exterior"]}
        )

        if self.variance_reject:
            variance_reject = dem_array.sel(masked="full").var(dim=["y", "x"]) > VAR_REJECT
            dates_to_drop = dem_array.sel(masked="full").where(variance_reject, drop=True).date

            drop_list = "\n".join([str(dat) for dat in dates_to_drop.data])
            logger.info(f"{len(dates_to_drop)} Dates with variance above the threshold {VAR_REJECT}: \n{drop_list}")

            # Filter the original DataArray based on the variance threshold
            dem_array = dem_array.sel(date=~dem_array.date.isin(dates_to_drop))

        # QA and sorting
        assert not stats.empty
        # check if they have the same CRS
        assert (stats["crs_epsg"][0] == stats["crs_epsg"]).all()
        dem_array = dem_array.sortby("date")
        return dem_array, stats

