"""

        SupraLens
Supraglacial Lake Analysis Tool

This module provides a command-line interface for analyzing supraglacial lake data using ArcticDEM data.
The tool allows users to specify a lake ID or a lake geometry file, and performs various analyses on the
DEM data for the specified lake, including filtering based on cross-correlation, calculating vertical offsets,
and estimating lake volumes. The tool also generates plots and reports of the analysis results.


Author: Matthias Franz-Josef Rötzer
Creation Date: 28.04.2024
Correspondence: s223783@dtu.dk, private: matthias1roetzer@gmail.com
: TODO
    - move plotting and data handling steps outside of this script
    - simplify the export of plots
    - add tests
"""

import argparse
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from loguru import logger
from matplotlib import pyplot as plt

import data
from data.geometries import load_lake_geometry_from_file
from utils.plotting import plot_heatmap
from utils.constants import PLOTTING_DIR, REPORT_FP, PLOT_QUALITY, CC_REJECT, RESULTS_DEFAULT_DIR, LAKE_VOLUMES_FP, \
    BUFFER_SIZE, REF_DATE, OFFSET_METHODS
from data.data_statistics import calculate_correlation_coefficient, get_error_statistics, build_difference_array, \
    estimate_vertical_offset
from data.data_loader import DataLoader
from utils import plotting
from utils.execution_utils import log_memory_details


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(description="Supraglacial Lake analysis tool")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--lake-id", help="Lake id from that one dataset", type=int)
    group.add_argument("--lake-geometry-fp", help="Lake geometry as a geopackage")
    parser.add_argument("--date", help="Date for analysis")
    parser.add_argument("--buffer-size",
                        help="Buffer distance around the lake geometry in meters", type=int,
                        default=BUFFER_SIZE
                        )
    parser.add_argument("--difference-matrix", help="Calculate and plot a difference matrix", action="store_true")
    # parser.add_argument("--download", help="Only query or also download", action="store_true")
    parser.add_argument("--plot", help="Enable the generation of output plots", action="store_true")
    parser.add_argument("--cc-reject", help="Cross correlation value to filter by", type=float, default=CC_REJECT)

    parser.add_argument("--offset-method", help="Method to find the vertical offset with", choices=OFFSET_METHODS,
                        default="md")
    parser.add_argument("--smoothing-window-size",
                        help="Smoothing by rolling mean of the DEM. Define the window size",
                        type=int)
    parser.add_argument("--output-dir", help="Output directory to write the results to", type=str,
                        default=RESULTS_DEFAULT_DIR)
    return parser.parse_args()


def main():
    """
    Main entry point for the Supraglacial Lake analysis tool.
    """
    args = parse_args()

    start_time = time.time()
    logger.info(f"Supraglacial Lake analysis".upper())
    logger.info(f"Received arguments: {args}")

    # Init args new
    lake_id: int = args.lake_id
    output_dir = Path(args.output_dir)

    cc_reject = args.cc_reject
    vertical_offset_method = args.offset_method
    logger.info(f"Output Directory: {output_dir.absolute()}")
    lake_geometry_fp: Path = Path(args.lake_geometry_fp) if args.lake_geometry_fp else None
    lake_geometry = load_lake_geometry_from_file(lake_geometry_fp) if lake_geometry_fp else None

    buffer_size: int = args.buffer_size
    logger.warning(f"Buffer-size not defined, using default of {BUFFER_SIZE=}m") if buffer_size == BUFFER_SIZE else None
    date: datetime.date = pd.to_datetime(args.date).date() if args.date else None
    if not lake_geometry_fp:
        logger.warning(f"Date not defined, using the union of all dates") if date is None else None

    lake_id = lake_id or lake_geometry_fp.stem
    logger.info(f"Using user input setting lake_id to {lake_id}")
    plot_data: bool = args.plot
    calc_difference_array: bool = args.difference_matrix
    smoothing_window_size = args.smoothing_window_size

    if plot_data:
        plotting_dir: Path = output_dir / Path(
            PLOTTING_DIR.format(lake_id=lake_id, date=date.strftime("%Y-%m-%d") if date else "all")
        )
        plotting_dir.mkdir(parents=True, exist_ok=True)
        plotting.init_plotting_params()
    else:
        plotting_dir = None

    # Start the analytics
    data_loader = DataLoader(
        lake_id=lake_id,
        buffer=buffer_size,
        output_dir=output_dir,
        date=date,
        lake_geometry=lake_geometry
    )
    # Filter based on Cross-correlation
    dem_da, stats = data_loader.load_dem_array()
    dem_da_cc = calculate_correlation_coefficient(data_array=dem_da.sel(masked="exterior"))

    cc_filter = abs(dem_da_cc[0]) > cc_reject  # first row indicates cross-correlation with tile (mosaic)
    logger.warning(
        f"Dropping {sum((~cc_filter).astype(int))} DEMs have a cross-correlation to the mosaic of > {cc_reject} "
        f"thus are likely incoherent"
    )
    dem_da = dem_da[:, cc_filter, :, :]
    if plot_data:
        date_list = dem_da["date"].values
        fig = plot_heatmap(
            dem_da_cc[cc_filter, :][:, cc_filter],
            title=f"Correlation Coefficient Matrix (>{cc_reject} to ref. mosaic)",
            heatmap_kwargs=dict(xticklabels=date_list, yticklabels=date_list, fmt=".2f")
        )
        fig.savefig(plotting_dir / "cross_correlation.jpg", dpi=PLOT_QUALITY)

    # Check DEM date integrity
    n_unique_dates = len(set(dem_da.coords["date"].values))
    n_all_dates = len(dem_da.coords["date"].values)
    if n_all_dates != n_unique_dates:
        logger.warning(
            "\nFound duplicate Dates for some DEMs. "
            f"\n\tDropped {n_all_dates - n_unique_dates} dates"
        )
    dem_da = dem_da.drop_duplicates(dim="date")
    if n_all_dates >= 50:
        logger.warning(f"Too many or too little dates (N={n_all_dates})")
    if n_all_dates == 1:
        raise ValueError("No DEMs to analyze. Maybe too much filtering?")

    if plot_data:
        logger.info(f"Generating Plots and exporting to \n{plotting_dir.absolute()}")

        profiles_fp = plotting_dir / "profiles_raw.jpg"
        vmin, vmax = dem_da.quantile(q=0.1), dem_da.quantile(q=0.9)  # making the imshow plots more robust to outliers
        # 1) Visualize the DEMs
        _ = dem_da.sel(masked="masked").plot.imshow(
            col="date",
            col_wrap=int(np.sqrt(n_all_dates)),  # multiplot settings
            # aspect=ds.dims["lon"] / ds.dims["lat"],  # for a sensible fig size
            # subplot_kws={"projection": map_proj},
            cbar_kwargs={"shrink": 0.5},
            vmin=vmin, vmax=vmax
        )
        facet_fp = plotting_dir / f"date{date if date else 's_all'}_lake_masked.jpg"
        plt.savefig(facet_fp, dpi=PLOT_QUALITY)

        # 2) plot profiles
        fig = plotting.plot_profiles(dem_da.sel(masked="full"))
        fig.savefig(profiles_fp, dpi=PLOT_QUALITY)

    # Smoothing
    if smoothing_window_size:
        logger.info(
            f"Applying smoothing of size {smoothing_window_size}x{smoothing_window_size}m to the entire dataset ")
        dem_da = dem_da.rolling(
            x=smoothing_window_size // 2, y=smoothing_window_size // 2,
            center=True).mean()

    # Split up dataset.
    tile_dem_da = dem_da.sel(date=REF_DATE)
    raw_strip_da = dem_da.drop_sel(date=REF_DATE)
    del dem_da

    # This is where the magic happens.
    dz = estimate_vertical_offset(
        reference_data_array=tile_dem_da.sel(masked="exterior"),
        to_transform_data_array=raw_strip_da.sel(masked="exterior"),
        method=vertical_offset_method
    )

    offsets = dz.to_pandas()
    offsets.name = "dz"
    logger.info(
        f"\nOffsets (dz): "
        f"\n{offsets}"
    )
    logger.info("Calculating error statistics")
    to_tile_error = get_error_statistics(
        reference_da=tile_dem_da.sel(masked="full"),
        compare_da=raw_strip_da.sel(masked="full")
    )
    transformed_da = raw_strip_da + dz
    del raw_strip_da
    transformation_error_statistic = get_error_statistics(
        reference_da=tile_dem_da.sel(masked="full"),
        compare_da=transformed_da.sel(masked="full")
    )

    # diff_da = build_difference_array(data_array=transformed_da.sel(masked="masked"))
    full_da = xr.concat([transformed_da, tile_dem_da], dim="date")
    full_da = full_da.sortby("date")
    del transformed_da, tile_dem_da
    log_memory_details(full_da.nbytes)

    if plot_data:
        # Profiles
        fig = plotting.plot_profiles(full_da.sel(masked="full"))
        profiles_fp = plotting_dir / "profiles_transformed.jpg"
        fig.savefig(profiles_fp, dpi=PLOT_QUALITY)

    if calc_difference_array:
        logger.info("Calculating difference array")
        if n_all_dates > 10:
            logger.warning(f"Calculating the difference array is unstable for more than 10 dates ({n_all_dates} given)")
        diff_da = build_difference_array(data_array=full_da.sel(masked="full"))
        # 3) Difference Matrix
        if calc_difference_array:
            diff_da.plot.pcolormesh(
                row="date_1",
                col="date_2",
                vmin=-2, vmax=2,
                cmap="RdBu_r",
                cbar_kwargs={"shrink": 0.5},
            )
            plotting_dir: Path = output_dir / Path(
                PLOTTING_DIR.format(lake_id=lake_id, date=date.strftime("%Y-%m-%d") if date else "all")
            )

            difference_matrix_fp = plotting_dir / "difference_matrix.jpg"
            plt.savefig(difference_matrix_fp, dpi=PLOT_QUALITY)

    else:
        # plotting the difference
        logger.info("Calculating and plotting the difference")
        diff_da = full_da - full_da.sel(date=REF_DATE)
        diff_da.sel(masked="full", date=~(diff_da.date == REF_DATE)).plot.pcolormesh(  # plot "full", ignore last date
            col="date",
            col_wrap=int(np.sqrt(n_all_dates)),
            vmin=-4, vmax=4,
            cmap="RdBu_r",
            cbar_kwargs={"shrink": 0.5},
        )
        if plot_data:
            plt.savefig(plotting_dir / "difference_to_mosaic.jpg", dpi=PLOT_QUALITY)

    logger.info("Finally Estimate the Lake Volume")
    lake_depth_raster = diff_da.where(diff_da < 0)
    volume = (lake_depth_raster * 2 * 2).sum(dim=("x", "y"))  # 2x2 meter resolution
    # logger.info(f"Lake Volume: {volume.values} m^3")
    volume_mask = np.tril(np.ones_like(volume, dtype=bool))
    # plot_profiles(diff_da["date_2"].rename(date_2="date"))
    if plotting_dir and calc_difference_array:
        # 4) Lakes Heatmap
        plotting.plot_heatmap(
            volume, title="Lake Volumes",
            heatmap_kwargs=dict(
                xticklabels=volume["date_1"].values,
                yticklabels=diff_da["date_2"].values,
                fmt=".2e",
                mask=volume_mask
            ),
        )
        estimated_lake_volumes_fp = plotting_dir / "estimated_lake_volumes.jpg"
        plt.savefig(estimated_lake_volumes_fp, dpi=PLOT_QUALITY)

    # logger.info(volume)
    report = pd.concat(
        # [offsets, transformation_error_statistic],
        [to_tile_error, transformation_error_statistic],
        axis=1, keys=["to_mosaic", "transformation"]
    )
    report.to_csv(Path(output_dir / REPORT_FP.format(lake_id=lake_id, date=date if date else "all")))
    volume.to_pandas().to_csv(Path(output_dir / LAKE_VOLUMES_FP.format(lake_id=lake_id, date=date if date else "all")))
    logger.info(f"Analysis Finished in {time.time() - start_time:.1f} seconds")


if __name__ == "__main__":
    main()
