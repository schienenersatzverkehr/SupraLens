"""
ArcticDEM Query and Retrieve Module

The module includes functions for:

* Querying DEM tiles and strips by lake ID and date
* Retrieving lake geometries by ID and date
* Downloading DEM data from file URLs
* Handling ArcticDEM-specific URL formatting and metadata
The module is designed to be used with the ArcticDEM AWS S3 Bucket
https://www.pgc.umn.edu/guides/stereo-derived-elevation-models/pgc-dem-products-arcticdem-rema-and-earthdem/#section-8

TODO:
    - use AWS SDK for data retrieval

"""
import argparse
import json
import urllib.request
import urllib.request
from typing import Union, List, Tuple
from urllib.parse import urlparse

import dask_geopandas as dgpd
import fiona
import geopandas as gpd
import requests
from loguru import logger
from shapely import intersects
from shapely.geometry import shape, Polygon
from shapely.ops import unary_union
from tqdm import tqdm

from utils.constants import *


def get_tiles_by_lake_id(
        lake_id: int,
        date_str: str,
        buffer: int,
        output_dir: Path,
        lake_geometry: gpd.GeoDataFrame = None,
) -> Tuple[Polygon, gpd.GeoDataFrame]:
    with fiona.open(TILE_INDEX_FP) as src:
        src_crs = src.crs
        if isinstance(lake_id, int):
            lake_query_hull = get_lake_hull_by_id(
                lake_id=lake_id,
                buffer=buffer,
                date_str=date_str,
                target_crs=src_crs,
                output_dir=output_dir
            )
        elif not lake_geometry.empty:
            lake_query_hull = lake_geometry.convex_hull.squeeze()
        else:
            raise NotImplementedError

        tile_query_gdf = gpd.GeoDataFrame.from_features([
            tile_feature
            for tile_feature in tqdm(src, desc="Querying Tile index...")
            if intersects(shape(tile_feature["geometry"]), lake_query_hull)
            # shape(tile_feature["geometry"]).intersects(lake_query_hull)
        ])
        if tile_query_gdf.empty:
            raise RuntimeError("No intersecting tiles found!")

    logger.info(f"\nLake geometry of size {lake_query_hull.area:.3f} "
                f"\nINTERSECTS with {tile_query_gdf.shape[0]} tile(s)"
                f"\nSource CRS: {src_crs}")
    if tile_query_gdf.shape[0] > 1:
        # :todo merge interesecting tiles
        # raise NotImplementedError("Not more than one tile currently")
        logger.warning("Handling more than one interesting tile is not implemented yet.")
    return lake_query_hull, tile_query_gdf


def get_strips_by_lake_id(
        lake_id: Union[int, str],
        date_str: str,
        buffer: int,
        output_dir: Path,
        # method: str = "contains",
        lake_geometry=None
) -> Tuple[Polygon, gpd.GeoDataFrame]:
    if isinstance(lake_id, int):  # reminder: if the id is a str value, it's user defined input
        lake_query_hull = get_lake_hull_by_id(lake_id=lake_id, buffer=buffer, date_str=date_str, output_dir=output_dir)
    else:
        lake_query_hull = (
            lake_geometry
            .buffer(buffer)
            .to_crs(WGS_CRS)
            .convex_hull
            .squeeze()
        )
    strip_query_data_fp = output_dir / Path(
        STRIP_QUERY_GEOMETRIES_FP.format(lake_id=lake_id, date=date_str, buffer=buffer))
    strip_query_data_fp.parent.mkdir(exist_ok=True, parents=True)
    logger.info("Query strip data")
    if strip_query_data_fp.exists():
        filtered_gdf = gpd.read_file(strip_query_data_fp)
    else:
        # spatial_operation = contains if method == "contains" else intersects
        dask_gdf = dgpd.read_file(STRIP_INDEX_FP, chunksize=1024 * 10)
        strip_query_gdf = dask_gdf[dask_gdf.contains(lake_query_hull)].compute()
        # with fiona.open(STRIPS_FP) as src:
        #     strip_query_gdf = gpd.GeoDataFrame.from_features([
        #         strip_feature
        #         for strip_feature in tqdm(src, desc="Querying Strip index...")
        #         if spatial_operation(shape(strip_feature["geometry"]), lake_query_hull)
        #     ])
        if strip_query_gdf.empty:
            raise ValueError(f"No strips found containing {lake_id=} fully")
        strip_query_gdf.set_crs(crs=WGS_CRS)

        # apply filtering:
        filtered_gdf = strip_query_gdf[strip_query_gdf["valid_area_percent"] > VALID_AREA_PERCENT]
        filtered_gdf = filtered_gdf[filtered_gdf["avg_expected_height_accuracy"] > AVG_EXPECTED_HEIGHT_ACCURACY]
        filtered_gdf = filtered_gdf[filtered_gdf["acqdate1"] > VALID_EARLIEST_DATE]
        logger.info(
            f"- Found {len(strip_query_gdf)} strips intersecting with the lake of interest"
            f"- Dropped {len(strip_query_gdf) - len(filtered_gdf)} strips"
            f"\nCriterion: "
            f"\nDEM > ({AVG_EXPECTED_HEIGHT_ACCURACY=})"
            f"\nDEM > ({VALID_AREA_PERCENT=})"
            f"\nDate > ({VALID_EARLIEST_DATE=})"
        )

        filtered_gdf.to_file(strip_query_data_fp, crs=WGS_CRS)

    return lake_query_hull, filtered_gdf


def execute_download_by_lake_id(lake_id: int, download: bool, date_str: str) -> None:
    """
    :param date_str:
    :param lake_id
    :param download todo remove
    """

    # lake_query_hull, tile_query_gdf = get_tiles_by_lake_id(lake_id, method="overlaps")
    _, strip_query_gdf = get_strips_by_lake_id(lake_id, method="contains", date_str=date_str)
    # tile_query_gdf["fileurl"]

    url_df = strip_query_gdf["fileurl"].apply(download_dem_from_fileurl, download=download)
    url_df.index = strip_query_gdf["fileurl"]
    # url_df.to_csv(STRIP_QUERY_DATA_FP.format(lake_id=lake_id)

    # strip_query_gdf["fileurl"].progress_apply(download_dem_from_fileurl)
    logger.info("Download finished")
    return


def get_lake_geometries_by_id(
        lake_id: int,
        date_str: str,
        output_dir: Path,
        supraglacial_fp: str = SUPRAGLACIAL_FP,
) -> gpd.GeoDataFrame:
    """
    takes 4 seconds longer than the fiona only loading method...
    :param output_dir:
    :param date_str:
    :param lake_id:
    :param supraglacial_fp:
    :return: GeoDataFrame containing all lakes with a certain id
    """
    lake_fp = output_dir / Path(LAKE_INDEX_FP.format(lake_id=lake_id, date=date_str))
    if not lake_fp.parent.exists():
        lake_fp.parent.mkdir(parents=True, exist_ok=False)

    if lake_fp.exists():
        lake_geometries = gpd.read_file(lake_fp)
    else:
        lakes_gdf = gpd.read_file(supraglacial_fp)
        lake_geometries = lakes_gdf[lakes_gdf["id1"] == lake_id]
        if lake_geometries.empty:
            raise ValueError(f"No lake polygons found matching ID {lake_id}")

    if date_str != "all":
        lake_geometries = lake_geometries[lake_geometries["date"] == date_str]
        if lake_geometries.empty:
            raise ValueError(f"No lake polygons found matching date {date_str}")

    lake_geometries.to_file(lake_fp)

    return lake_geometries


def get_lake_hull_by_id(
        lake_id: int,
        date_str: str,
        buffer: int,
        output_dir: Path,
        supraglacial_fp: str = SUPRAGLACIAL_FP,
        target_crs=WGS_CRS
) -> Polygon:
    """ loads a geo-dataset containing lake geometries
    Load the geo-dataset containing supraglacial lake geometries
    and returns a lake's surrounding hull as a Polygon in the target CRS.
    :param output_dir:
    :param date_str:
    :param target_crs:
    :todo load data from link, not local
    :param buffer:
    :param lake_id: The ID of the lake to retrieve
    :param supraglacial_fp: The file path to the supraglacial lakes dataset
    (:param target_crs: The target CRS for the lake geometry) going with WGS as a standard
    :return: A lake's surrounding hull as a Polygon in the target CRS
    """

    lakes_geometries = get_lake_geometries_by_id(
        lake_id, date_str, supraglacial_fp=supraglacial_fp, output_dir=output_dir
    )
    buffered_lakes_geometries = lakes_geometries.buffer(buffer)
    if buffered_lakes_geometries.crs.to_epsg() != target_crs.to_epsg():
        logger.warning(f"Reprojected lake polygons to {target_crs}")
        lakes_geometries = buffered_lakes_geometries.to_crs(target_crs)

    lake_unary = unary_union(lakes_geometries.geometry)
    lake_hull = lake_unary.convex_hull
    # lake_hull = lake_unary.concave_hull

    assert isinstance(lake_hull, Polygon)
    assert lake_hull.is_valid
    assert lake_unary.is_valid
    return lake_hull


def _get_lake_geometries_by_id(lake_id: int, supraglacial_fp: str, target_crs=WGS_CRS) -> pd.Series:
    """ deprecated """
    with fiona.open(supraglacial_fp) as src:
        filtered_by_id = list(
            filter(lambda f: (f["properties"]["id1"] == lake_id), src)
        )
        if len(filtered_by_id) == 0:
            raise ValueError("No polygons found matching this ID")
        src_crs = src.crs
        lake_polygon_series = gpd.GeoDataFrame(
            geometry=[shape(poly["geometry"]) for poly in filtered_by_id],
            crs=src_crs
        )
    if src_crs != target_crs:
        logger.warning(f"Reprojected lake polygons to {target_crs}")
        lake_polygon_series = lake_polygon_series.to_crs(target_crs)
    return lake_polygon_series


def download_files_from_list(urls: List, download_dir: Union[Path, str]) -> None:
    """"""
    for url in urls:
        response = requests.get(url, stream=True)  #:todo wrap with try catch
        url = Path(url) if isinstance(url, str) else url
        filename = download_dir / url.name
        if filename.exists():
            # logger.info(f"skipping download of {filename}")
            pass
        with open(filename, "wb") as file:
            for chunk in tqdm(response.iter_content(chunk_size=1024 * 4), desc=f"{url}"):
                if chunk:
                    file.write(chunk)


def get_dem_tif_url_from_fileurl(file_url: str) -> Path:
    """ uses some ArcticDEM specific url handling """
    path_edit = file_url.split("https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/")[1]
    path_edit = path_edit.replace("mosaic", "mosaics")  # not ideal
    dem_tif_url = urlparse(TARGET_URL + path_edit.replace(".tar.gz", "_dem.tif"))
    return Path(dem_tif_url.geturl())


def download_dem_from_fileurl(file_url: str, download: bool = False) -> pd.Series:
    """
    Download DEM data from a file URL and return a pandas Series with the downloaded file URLs.
    very specific formatting happening here ...
    :param file_url: The URL of the DEM file to download
    :param download: Whether to download the data or just query it
    :return: A pandas Series with the downloaded file URLs
    """
    path_edit = file_url.split("https://data.pgc.umn.edu/elev/dem/setsm/ArcticDEM/")[1]
    meta_s3_url = urlparse(TARGET_URL + path_edit.replace(".tar.gz", ".json"))

    with urllib.request.urlopen(meta_s3_url.geturl()) as url:
        data = url.read().decode("utf-8")
        strip_meta_dict = json.loads(data)

    to_download_file_ends = ["_dem.tif", "_bitmask.tif", "_matchtag.tif", "_mdf.txt"]
    to_download_file_ends = ["_dem.tif", "_mdf.txt"]
    to_download_urls = [
        meta_s3_url.geturl().replace(".json", file_end)
        for file_end in to_download_file_ends
    ]

    download_dir = Path(meta_s3_url.path[1:]).parent
    download_dir.mkdir(parents=True, exist_ok=True)
    if download:
        download_files_from_list(urls=to_download_urls, download_dir=download_dir)
    # contains the same info as mdf.txt
    # write_json(strip_meta_dict, json_file_path=download_dir / f"{meta_s3_url}_meta.json")

    # gather a collection to return
    to_download_file_ends.append("avg_expected_height_accuracy")
    to_download_file_ends.append("rmse")

    to_download_urls.append(strip_meta_dict["properties"]["pgc:avg_expected_height_accuracy"])
    to_download_urls.append(strip_meta_dict["properties"]["pgc:rmse"])

    return pd.Series(to_download_urls, index=to_download_file_ends)


def write_json(to_write: dict, json_file_path: Union[str, Path]):
    json_file_path = str(json_file_path) if isinstance(json_file_path, Path) else json_file_path
    with open(json_file_path, "w") as json_file:
        json.dump(to_write, json_file)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lake-id", help="Lake id from that one dataset")
    parser.add_argument("--download", help="only query or also download", action="store_true")
    args = parser.parse_args()

    lake_id = int(args.lake_id)
    # lake_id = 191587  # candidates: 191587, 180359, 193456
    # execute_download_by_lake_id(lake_id, download=args.download)
    raise NotImplementedError


def query_feature_service(lake_query_hull: Polygon, src_crs):
    """ not working :todo"""
    feature_service_url = "https://services.arcgis.com/8df8p0NlLFEShl0r/arcgis/rest/services/PGC_ArcticDEM_Strip_Index/FeatureServer"
    layer_id = 0
    ring = [[list(x) for x in lake_query_hull.exterior.coords]]
    params = {
        "f": "json",  # Return results in JSON format
        "where": "1=1",  # Simple query to return all features
        "outFields": "*",  # Return all fields
        "geometryType": "esriGeometryPolygon",  # Return only polygon features
        "spatialRel": "esriSpatialRelIntersects",  # Return features that intersect with the given polygon
        "geometry": json.dumps({
            "rings": ring,
            "spatialReference": {"wkid": src_crs.to_epsg()}  # Define the spatial reference of the given polygon
        })
    }
    query_url = f"{feature_service_url}/{layer_id}/query"
    response = requests.get(query_url, params=params)
    if response.status_code == 200:
        data = response.json()
        logger.info("")
        return data
    else:
        logger.info("Error:", response.status_code)
        return


if __name__ == "__main__":
    main()
