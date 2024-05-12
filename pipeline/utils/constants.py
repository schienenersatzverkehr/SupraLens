from pathlib import Path

import pandas as pd
from pyproj import CRS

# Paths
SUPRAGLACIAL_FP = Path("./assets/jakobshavn_supraglacial_lakes.gpkg")
STRIP_INDEX_FP = Path("./assets/ArcticDEM_Strip_Index_s2s041_gpkg.gpkg")
TILE_INDEX_FP = Path("./assets/ArcticDEM_Mosaic_Index_v4_1_gpkg.gpkg")

RESULTS_DEFAULT_DIR = "./results/"

LAKE_INDEX_FP = "./lake_id_{lake_id}_date_{date}/geometry_date_{date}.gpkg"
STRIP_QUERY_GEOMETRIES_FP = "./lake_id_{lake_id}_date_{date}/overlapping_strips_buffer_{buffer}.gpkg"
STRIP_QUERY_DATA_FP = "./lake_id_{lake_id}_date_{date}/overlapping_strips_buffer_{buffer}.csv"
PLOTTING_DIR = "./lake_id_{lake_id}_date_{date}/plots"

REPORT_FP = "./lake_id_{lake_id}_date_{date}/{lake_id}_report.csv"
LAKE_VOLUMES_FP = "./lake_id_{lake_id}_date_{date}/{lake_id}_estimated_lake_volumes.csv"

TARGET_URL = "https://pgc-opendata-dems.s3.us-west-2.amazonaws.com/arcticdem/"

BUFFER_SIZE = 200  # standard buffer size in m
REF_DATE = pd.to_datetime("2001-01-01").date()

PLOT_QUALITY = 250  # dpi
OFFSET_METHODS = ["rmse", "md", "linear"]

WGS_CRS = CRS.from_epsg(4326)

# Arctic DEM data filters within dataloader
VALID_AREA_PERCENT = 0.5
AVG_EXPECTED_HEIGHT_ACCURACY = 0
VALID_EARLIEST_DATE = pd.Timestamp("2013-01-01")
VAR_REJECT = 10000

CC_REJECT = 0.  # cross correlation reject
