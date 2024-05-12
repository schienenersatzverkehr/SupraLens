from typing import Optional

import numpy as np
import xarray as xr
import seaborn as sns

from matplotlib import pyplot as plt

import matplotlib


def init_plotting_params():
    matplotlib.rcParams['mathtext.fontset'] = 'stix'
    matplotlib.rcParams['font.family'] = 'STIXGeneral'
    matplotlib.pyplot.title(r'ABC123 vs $\mathrm{ABC123}^{123}$')


def plot_profiles(dem_data_array: xr.DataArray, mask: Optional[xr.DataArray] = xr.DataArray()) -> plt.Figure:
    ARROW_PROPS = {
        "arrowstyle": "->",
        "connectionstyle": "arc3,rad=0",
        "color": "red"
    }
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    # Set the arrow properties

    raster_width, raster_height = dem_data_array.shape[1], dem_data_array.shape[2]
    # Add the arrow
    ax[0].annotate(
        '', xy=(raster_height // 2, raster_width),
        xytext=(raster_height // 2, 0),
        arrowprops=ARROW_PROPS
    )
    ax[0].annotate(
        '', xy=(raster_height, raster_width // 2,),
        xytext=(0, raster_width // 2),
        arrowprops=ARROW_PROPS
    )

    # Add the textbox
    ax[0].text((raster_height // 2), (raster_width // 10), "(A)", ha="center", fontsize=24, color="r")
    ax[0].text((raster_height // 10), (raster_width // 2), "(B)", ha="center", fontsize=24, color="r")

    # x, y = shp.exterior.xy
    # img = ax[0].plot(x, y)
    ax[0].imshow(dem_data_array[0])  # just plot the first one
    if len(mask.shape) == 2:
        ax[0].imshow(mask, alpha=0.1)
    ax[0].set_title(f"{list(dem_data_array.indexes.values())[0][0]}")
    _ = (dem_data_array.isel(x=raster_height // 2)
         .plot(hue="date", add_legend=False, ax=ax[1]))
    _ = (dem_data_array.isel(y=raster_width // 2)
         .plot(hue="date", add_legend=False, ax=ax[2]))

    ax[1].set_title("Profile (A)")
    ax[2].set_title("Profile (B)")
    plt.legend(dem_data_array["date"].values, loc="center left", bbox_to_anchor=(1, 0.5))
    ax[1].grid()
    ax[2].grid()
    return fig


def plot_cross_difference(data_array: xr.DataArray) -> plt.Figure:
    """ deprecated """
    assert len(data_array.dims) == 3

    dates = data_array["date"].values
    n_dems = len(dates)
    fig, ax = plt.subplots(n_dems, n_dems, figsize=(15, 15))
    for j in range(n_dems):
        for i in range(n_dems):
            if i == j:
                im = ax[i, j].imshow(data_array[i])
                ax[i, j].set_title(f"{dates[i]}")
            else:
                im = ax[i, j].imshow(data_array[i] - data_array[j])
                # ax[i, j].set_title(f"{dem_corr_coef[i, j]:.3f}")
            ax[i, j].tick_params(left=False, right=False, labelleft=False,
                                 labelbottom=False, bottom=False)
    # cbar = fig.colorbar(ax[-1, -1].imshow(normalized_dem[i] - normalized_dem[j]), ax=ax, shrink=0.3)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax, shrink=0.6)
    # fig.subplots_adjust(right=0.8)
    # plt.margins(tight=True)
    # cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    # fig.colorbar(im, cax=cbar_ax, shrink=0.6)

    # fig.colorbar(im, ax=ax.ravel().tolist())
    fig.tight_layout()
    return fig


def plot_heatmap(even_matrix: np.array, title: str, **heatmap_kwargs) -> plt.Figure:
    """
    Plots a NxN matrix of values as a heatmap e.g. a correlation coefficient matrix.
    :param: even_matrix (numpy array): e.g. 8x8 correlation coefficient matrix
    :returns: matplotlib.pyplot.Axes: The plot object
    """
    # if not isinstance(even_matrix, np.ndarray):
    #     raise ValueError("Input must be a numpy array")
    fig, ax = plt.subplots(figsize=even_matrix.shape)
    sns.heatmap(
        even_matrix,
        annot=True, cmap="coolwarm",
        square=True, ax=ax,
        cbar_kws={"shrink": 0.5},
        **heatmap_kwargs["heatmap_kwargs"]
    )
    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Date")

    return fig
