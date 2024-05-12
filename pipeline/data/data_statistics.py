import pandas as pd
from dask import delayed
from typing import Any
import numpy as np
import xarray as xr
from loguru import logger


def get_error_statistics(reference_da: xr.DataArray, compare_da: xr.DataArray):
    """
    Calculates error metrics of two data arrays of equal (x,y)-size.
    :param: reference_da (xr.DataArray): Reference data array.
    :param: compare_da (xr.DataArray): Data array to compare with the reference
    :returns: pd.DataFrame: A DataFrame containing PSNR, RMSE, and MAE error statistics.

    Notes:
    - The function can handle two data arrays with the same date size or only one reference date.
    - The data arrays must have the same shape, otherwise, the reference data array will
        be broadcast to match the shape of the compare data array.
    """
    if reference_da.shape != compare_da.shape:
        reference_da.broadcast_like(compare_da)
    difference = reference_da - compare_da
    rmse = np.sqrt((difference ** 2).mean(["x", "y"]))
    mae = abs(difference).mean(["x", "y"])
    max_val = np.max(compare_da)
    psnr = 20 * np.log10(max_val / rmse)
    return pd.DataFrame(
        [psnr.values, rmse.values, mae.values],
        index=["PSNR", "RMSE", "MAE"],
        columns=compare_da.indexes["date"]
    ).transpose()


def calculate_correlation_coefficient(data_array: xr.DataArray) -> Any | None:
    """
    Calculate the correlation coefficient matrix from a 3D DataArray.
    :param: data_array (xr.DataArray): 3D DataArray containing the data to calculate the correlation coefficient matrix from.
    :param: plot (bool, optional): Whether to plot the correlation coefficient matrix. Defaults to False.

    :returns np.ndarray: The correlation coefficient matrix.
    """
    n_dems = data_array.shape[0]
    reshaped_data = data_array.to_numpy().reshape(n_dems, -1)  # full rasters
    masked = np.ma.masked_invalid(reshaped_data)
    dem_corr_coef = np.ma.corrcoef(masked).data

    return dem_corr_coef


def build_difference_array(data_array: xr.DataArray) -> xr.DataArray:
    """
    Builds a difference array from the input data array.
    :param: data_array (xr.DataArray): Input data array
    :returns: xr.DataArray: Difference array with dimensions ('date_1', 'date_2', 'y', 'x')

    """
    n_dems = len(data_array)
    diff_da = xr.DataArray(
        np.zeros((n_dems, *data_array.shape)),
        # np.zeros(), shape=(n_dems, n_dems, *data_array.shape[1:]),
        dims=['date_1', 'date_2', 'y', 'x'],
        coords={"date_1": data_array["date"].values, "date_2": data_array["date"].values}
    )

    # for i in tqdm(range(n_dems), desc="Building difference array"):
    #     for j in range(n_dems):
    #         if i != j:
    #             # diff_da.values[i, j, :, :] = data_array.values[i, :, :] - data_array.values[j, :, :]
    #             diff_da.values[i, j, :, :] = delayed(compute_diff)(data_array.isel(date=i), data_array.isel(date=j))

    diff_da.values = data_array.values[:, None, :, :] - data_array.values[None, :, :, :]
    np.triu(diff_da.values, k=1)[:] = diff_da.values
    # np.fill_diagonal(diff_da.values, data_array.values)
    return diff_da


@delayed
def compute_diff(self, da1: xr.DataArray, da2: xr.DataArray) -> np.ndarray:
    """Computes the difference between two data arrays for dask
    """
    return da1.values - da2.values


def estimate_vertical_offset(
        reference_data_array: xr.DataArray, to_transform_data_array: xr.DataArray, method: str = "md"
) -> xr.DataArray:
    """
    Assess the vertical accuracy of Digital Elevation Models in the vertical by different statistical co-regsitration methods
    :param to_transform_data_array:
    :param reference_data_array:
    :param method: vertical offset estimation method
    "md" mean difference
    "rmse" root-mean-square error
    "linear" Linear Error at Confidence Levels.

    """
    diff = reference_data_array - to_transform_data_array
    if method == "md":
        dz = diff.mean(dim=["x", "y"])
    elif method == "rmse":
        dz = np.sqrt(diff.mean(dim=["x", "y"]) ** 2)
    elif method == "linear":
        logger.warning("Linear Error at Confidence Levels is not well tested")
        try:
            from scipy.stats import linregress
        except ImportError:
            logger.error("scipy.stats module not found. Linear Error at Confidence Levels method is not available.")

        dz = []
        for to_transform_date in to_transform_data_array:
            ref_flat = reference_data_array.values.flatten()
            to_transform_flat = to_transform_date.values.flatten()

            to_transform_flat = to_transform_flat[np.logical_not(np.isnan(to_transform_flat))]
            ref_flat = ref_flat[np.logical_not(np.isnan(ref_flat))]

            _, intercept, *_ = linregress(to_transform_flat, ref_flat)
            dz.append(intercept)
    else:
        raise ValueError("Wrong choice")
    return dz
