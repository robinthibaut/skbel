#  Copyright (c) 2021. Robin Thibaut, Ghent University

import os
import string
from os.path import join as jp

import joblib
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array, deprecated

from skbel.examples.demo_config import Setup
from skbel import utils
from skbel.goggles import explained_variance, _proxy_annotate, _proxy_legend, pca_scores
from skbel.spatial import (
    contours_vertices,
    grid_parameters,
    refine_machine,
)


def binary_polygon(
    xys: np.array,
    nrow: int,
    ncol: int,
    pzs: np.array,
    outside: float = -1,
    inside: float = 1,
) -> np.array:
    """
    Given a polygon whose vertices are given by the array pzs, and a matrix of
    centroids coordinates of the surface discretization, assigns to the matrix a certain value
    whether the cell is inside or outside said polygon.

    To compute the signed distance function, we need a negative/positive value.

    :param xys: Centroids of a grid' cells
    :param nrow: Number of rows
    :param ncol: Number of columns
    :param pzs: Array of ordered vertices coordinates of a polygon.
    :param pzs: Polygon vertices (v, 2)
    :param outside: Value to assign to the matrix outside of the polygon
    :param inside: Value to assign to the matrix inside of the polygon
    :return: phi = the binary matrix
    """

    # Creates a Polygon abject out of the polygon vertices in pzs
    poly = Polygon(pzs, True)
    # Checks which points are enclosed by polygon.
    ind = np.nonzero(poly.contains_points(xys))[0]
    phi = np.ones((nrow, ncol)) * outside  # SD - create matrix of 'outside'
    phi = phi.reshape((nrow * ncol))  # Flatten to have same dimension as 'ind'
    phi[ind] = inside  # Points inside the WHPA are assigned a value of 'inside'
    phi = phi.reshape((nrow, ncol))  # Reshape

    return phi


def binary_stack(xys: np.array, nrow: int, ncol: int, vertices: np.array) -> np.array:
    """
    Takes WHPA vertices and 'binarizes' the image (e.g. 1 inside, 0 outside WHPA).
    """
    # Create binary images of WHPA stored in bin_whpa
    bin_whpa = [
        binary_polygon(xys, nrow, ncol, pzs=p, inside=1, outside=-1) for p in vertices
    ]
    big_sum = np.sum(bin_whpa, axis=0)  # Stack them
    # Scale from 0 to 1
    big_sum -= np.min(big_sum)
    big_sum /= np.max(big_sum)
    return big_sum


def reload_trained_model(base_dir: str, root: str = None, well: str = None):
    if root is None:
        root = ""
    if well is None:
        well = ""

    res_dir = jp(base_dir, root, well, "obj")

    bel = joblib.load(jp(res_dir, "bel.pkl"))

    return bel


def whpa_plot(
    grf: float = None,
    well_comb: list = None,
    whpa: np.array = None,
    alpha: float = 0.4,
    halpha: float = None,
    lw: float = 0.5,
    bkg_field_array: np.array = None,
    vmin: float = None,
    vmax: float = None,
    x_lim: list = None,
    y_lim: list = None,
    xlabel: str = None,
    ylabel: str = None,
    cb_title: str = None,
    labelsize: float = 5,
    cmap: str = "coolwarm",
    color: str = "white",
    grid: bool = True,
    show_wells: bool = False,
    well_ids: list = None,
    title: str = None,
    annotation: list = None,
    fig_file: str = None,
    highlight: bool = False,
    show: bool = False,
):
    """
    Produces the WHPA plot, i.e. the zero-contour of the signed distance array.

    :param grid:
    :param grf: Grid cell size
    :param well_comb: List of well combination
    :param highlight: Boolean to display lines on top of filling between contours or not.
    :param annotation: List of annotations (str)
    :param xlabel:
    :param ylabel:
    :param cb_title:
    :param well_ids:
    :param labelsize: Label size
    :param title: str: plot title
    :param show_wells: bool: whether to plot well coordinates or not
    :param cmap: str: colormap name for the background array
    :param vmax: float: max value to plot for the background array
    :param vmin: float: max value to plot for the background array
    :param bkg_field_array: np.array: 2D array whose values will be plotted on the grid
    :param whpa: np.array: Array containing grids of values whose 0 contour will be computed and plotted
    :param alpha: float: opacity of the 0 contour lines
    :param halpha: Alpha value for line plots if highlight is True
    :param lw: float: Line width
    :param color: str: Line color
    :param fig_file: str:
    :param show: bool:
    :param x_lim: [x_min, x_max] For the figure
    :param y_lim: [y_min, y_max] For the figure
    """

    # Get basic settings
    focus = Setup.Focus
    wells = Setup.Wells

    if well_comb is not None:
        wells.combination = well_comb

    if y_lim is None:
        ylim = focus.y_range
    else:
        ylim = y_lim
    if x_lim is None:
        xlim = focus.x_range
    else:
        xlim = x_lim
    if grf is None:
        grf = focus.cell_dim
    else:
        grf = grf

    nrow, ncol, x, y = refine_machine(focus.x_range, focus.y_range, grf)

    # Plot background
    if bkg_field_array is not None:
        plt.imshow(
            bkg_field_array,
            extent=np.concatenate([focus.x_range, focus.y_range]),
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
        )
        cb = plt.colorbar(fraction=0.046, pad=0.04)
        cb.ax.set_title(cb_title)

    if halpha is None:
        halpha = alpha

    # Plot results
    if whpa is None:
        whpa = []

    if whpa.ndim > 2:  # New approach is to plot filled contours
        new_grf = 1  # Refine grid
        _, _, new_x, new_y = refine_machine(xlim, ylim, new_grf=new_grf)
        xys, nrow, ncol = grid_parameters(x_lim=xlim, y_lim=ylim, grf=new_grf)
        vertices = contours_vertices(x=x, y=y, arrays=whpa)
        b_low = binary_stack(xys=xys, nrow=nrow, ncol=ncol, vertices=vertices)
        contour = plt.contourf(
            new_x,
            new_y,
            1 - b_low,  # Trick to be able to fill contours
            # Use machine epsilon
            [np.finfo(float).eps, 1 - np.finfo(float).eps],
            colors=color,
            alpha=alpha,
        )
        if highlight:  # Also display curves
            for z in whpa:
                contour = plt.contour(
                    z,
                    [0],
                    extent=(xlim[0], xlim[1], ylim[0], ylim[1]),
                    colors=color,
                    linewidths=lw,
                    alpha=halpha,
                )

    else:  # If only one WHPA to display
        contour = plt.contour(
            whpa,
            [0],
            extent=np.concatenate([focus.x_range, focus.y_range]),
            colors=color,
            linewidths=lw,
            alpha=halpha,
        )

    # Grid
    if grid:
        plt.grid(color="c", linestyle="-", linewidth=0.5, alpha=0.2)

    # Plot wells
    well_legend = None
    if show_wells:
        plot_wells(wells, well_ids=well_ids, markersize=7)
        well_legend = plt.legend(fontsize=11)

    # Plot limits
    if x_lim is None:
        plt.xlim(xlim[0], xlim[1])
    else:
        plt.xlim(x_lim[0], x_lim[1])
    if y_lim is None:
        plt.ylim(ylim[0], ylim[1])
    else:
        plt.ylim(y_lim[0], y_lim[1])

    if title:
        plt.title(title)

    plt.xlabel(xlabel, fontsize=labelsize)
    plt.ylabel(ylabel, fontsize=labelsize)

    # Tick size
    plt.tick_params(labelsize=labelsize)

    if annotation:
        legend = _proxy_annotate(annotation=annotation, fz=14, loc=2)
        plt.gca().add_artist(legend)

    if fig_file:
        utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, bbox_inches="tight", dpi=300, transparent=False)
        plt.close()
    if show:
        plt.show()
        plt.close()

    return contour, well_legend


@deprecated()
def post_examination(
    root: str, xlim: list = None, ylim: list = None, show: bool = False
):
    focus = Setup.Focus()
    if xlim is None:
        xlim = focus.x_range
    if ylim is None:
        ylim = focus.y_range  # [335, 700]
    md = Setup.Directories()
    ndir = jp(md.forecasts_dir, "base", "roots_whpa", f"{root}.npy")
    sdir = os.path.dirname(ndir)
    nn = np.load(ndir)
    whpa_plot(
        whpa=nn,
        x_lim=xlim,
        y_lim=ylim,
        labelsize=11,
        alpha=1,
        xlabel="X(m)",
        ylabel="Y(m)",
        cb_title="SD(m)",
        annotation=["B"],
        bkg_field_array=np.flipud(nn[0]),
        color="black",
        cmap="coolwarm",
    )

    # legend = proxy_annotate(annotation=['B'], loc=2, fz=14)
    # plt.gca().add_artist(legend)

    plt.savefig(
        jp(sdir, f"{root}_SD.png"), dpi=300, bbox_inches="tight", transparent=False
    )
    if show:
        plt.show()
    plt.close()


def h_pca_inverse_plot(bel, fig_dir: str = None, show: bool = False):
    """
    Plot used to compare the reproduction of the original physical space after PCA transformation
    :param bel: BEL model
    :param fig_dir: str:
    :param show: bool:
    :return:
    """

    shape = bel.Y_shape

    if bel.Y_obs_pc is not None:
        v_pc = check_array(bel.Y_obs_pc.reshape(1, -1))
    else:
        try:
            Y_obs = check_array(bel.Y_obs, allow_nd=True)
        except ValueError:
            Y_obs = check_array(bel.Y_obs.to_numpy().reshape(1, -1))

        v_pc = bel.Y_pre_processing.transform(Y_obs)[
            :, : Setup.HyperParameters.n_pc_target
        ]

    nc = bel.Y_pre_processing["pca"].n_components_
    dummy = np.zeros((1, nc))
    dummy[:, : v_pc.shape[1]] = v_pc
    v_pred = bel.Y_pre_processing.inverse_transform(dummy)

    h_to_plot = np.copy(Y_obs.reshape(1, shape[1], shape[2]))

    whpa_plot(whpa=h_to_plot, color="red", alpha=1, lw=2)

    whpa_plot(
        whpa=v_pred.reshape(1, shape[1], shape[2]),
        color="blue",
        alpha=1,
        lw=2,
        labelsize=11,
        xlabel="X(m)",
        ylabel="Y(m)",
        x_lim=[850, 1100],
        y_lim=[350, 650],
    )

    # Add title inside the box
    an = ["B"]

    legend_a = _proxy_annotate(annotation=an, loc=2, fz=14)

    _proxy_legend(
        legend1=legend_a,
        colors=["red", "blue"],
        labels=["Physical", "Back transformed"],
        marker=["-", "-"],
    )

    if fig_dir is not None:
        utils.dirmaker(fig_dir)
        plt.savefig(
            jp(fig_dir, f"h_pca_inverse_transform.png"), dpi=300, transparent=False
        )
        plt.close()

    if show:
        plt.show()
        plt.close()


def plot_results(
    bel,
    d: bool = True,
    h: bool = True,
    root: str = None,
    base_dir: str = None,
    folder: str = None,
    annotation: list = None,
):
    """
    Plots forecasts results in the 'uq' folder
    :param bel: BEL model
    :param annotation: List of annotations
    :param h: Boolean to plot target or not
    :param d: Boolean to plot predictor or not
    :param root: str: Forward ID
    :param base_dir:
    :param folder: str: Well combination. '123456', '1'...
    :return:
    """
    if root is None:
        root = ""
    if folder is None:
        folder = ""
    # Directory
    md = jp(base_dir, root, folder)
    # Wells
    wells = Setup.Wells
    wells_id = list(wells.wells_data.keys())
    cols = [wells.wells_data[w]["color"] for w in wells_id if "pumping" not in w]

    if d:
        # Curves - d
        # Plot curves
        sdir = jp(md, "data")

        X = check_array(bel.X, allow_nd=True)
        try:
            X_obs = check_array(bel.X_obs, allow_nd=True)
        except ValueError:
            try:
                X_obs = check_array(bel.X_obs.to_numpy().reshape(1, -1))
            except AttributeError:
                X_obs = check_array(bel.X_obs.reshape(1, -1))

        tc = X.reshape((Setup.HyperParameters.n_posts,) + bel.X_shape)
        tcp = X_obs.reshape((-1,) + bel.X_shape)
        tc = np.concatenate((tc, tcp), axis=0)

        # Plot parameters for predictor
        xlabel = "Observation index number"
        ylabel = "Concentration ($g/m^{3})$"
        factor = 1000
        labelsize = 11

        curves(
            cols=cols,
            tc=tc,
            sdir=sdir,
            xlabel=xlabel,
            ylabel=ylabel,
            factor=factor,
            labelsize=labelsize,
            highlight=[len(tc) - 1],
        )

        curves(
            cols=cols,
            tc=tc,
            sdir=sdir,
            xlabel=xlabel,
            ylabel=ylabel,
            factor=factor,
            labelsize=labelsize,
            highlight=[len(tc) - 1],
            ghost=True,
            title="curves_ghost",
        )

        curves_i(
            cols=cols,
            tc=tc,
            xlabel=xlabel,
            ylabel=ylabel,
            factor=factor,
            labelsize=labelsize,
            sdir=sdir,
            highlight=[len(tc) - 1],
        )

    if h:
        # WHP - h test + training
        fig_dir = jp(base_dir, root)
        ff = jp(fig_dir, "whpa.png")  # figure name
        Y = check_array(bel.Y, allow_nd=True)
        try:
            Y_obs = check_array(bel.Y_obs, allow_nd=True)
        except ValueError:
            Y_obs = check_array(bel.Y_obs.to_numpy().reshape(1, -1))
        h_test = Y_obs.reshape((bel.Y_shape[1], bel.Y_shape[2]))
        h_training = Y.reshape((-1,) + (bel.Y_shape[1], bel.Y_shape[2]))
        # Plots target training + prediction
        whpa_plot(whpa=h_training, color="blue", alpha=0.5)
        whpa_plot(
            whpa=h_test,
            color="r",
            lw=2,
            alpha=0.8,
            xlabel="X(m)",
            ylabel="Y(m)",
            labelsize=11,
        )
        colors = ["blue", "red"]
        labels = ["Training", "Test"]
        legend = _proxy_annotate(annotation=[], loc=2, fz=14)
        _proxy_legend(legend1=legend, colors=colors, labels=labels, fig_file=ff)
        plt.close()

        # WHPs
        ff = jp(md, "uq", f"{root}_cca_{bel.cca.n_components}.png")
        forecast_posterior = bel.random_sample(n_posts=Setup.HyperParameters.n_posts)
        forecast_posterior = bel.inverse_transform(forecast_posterior)
        forecast_posterior = forecast_posterior.reshape(
            (-1,) + (bel.Y_shape[1], bel.Y_shape[2])
        )

        # I display here the prior h behind the forecasts sampled from the posterior.
        well_ids = [0] + list(map(int, list(folder)))
        labels = ["Training", "Samples", "True test"]
        colors = ["darkblue", "darkred", "k"]

        # Training
        _, well_legend = whpa_plot(
            whpa=h_training,
            alpha=0.5,
            lw=0.5,
            color=colors[0],
            show_wells=True,
            well_ids=well_ids,
            show=False,
        )

        # Samples
        whpa_plot(
            whpa=forecast_posterior,
            color=colors[1],
            lw=1,
            alpha=0.8,
            highlight=True,
            show=False,
        )

        # True test
        whpa_plot(
            whpa=h_test,
            color=colors[2],
            lw=0.8,
            alpha=1,
            x_lim=[800, 1200],
            xlabel="X(m)",
            ylabel="Y(m)",
            labelsize=11,
            show=False,
        )

        # Other tricky operation to add annotation
        legend_an = _proxy_annotate(annotation=annotation, loc=2, fz=14)

        # Tricky operation to add a second legend:
        _proxy_legend(
            legend1=well_legend,
            extra=[legend_an],
            colors=colors,
            labels=labels,
            fig_file=ff,
        )


def mode_histo(
    colors: list,
    an_i: int,
    wm: np.array,
    title: str = None,
    fig_name: str = "average",
    directory: str = None,
):
    """

    :param directory:
    :param title:
    :param colors:
    :param an_i: Figure annotation
    :param wm: Arrays of metric
    :param fig_name:
    :return:
    """
    if directory is None:
        directory = Setup.Directories.forecasts_dir
    alphabet = string.ascii_uppercase
    wid = list(map(str, Setup.Wells.combination))  # Wel identifiers (n)

    pipeline = Pipeline(
        [
            ("s_scaler", StandardScaler()),
        ]
    )
    wm = pipeline.fit_transform(wm)

    modes = []  # Get MHD corresponding to each well's mode
    for i, m in enumerate(wm):  # For each well, look up its MHD distribution
        count, values = np.histogram(m, bins="fd")
        # (Freedman Diaconis Estimator)
        # Robust (resilient to outliers) estimator that takes into account data variability and data size.
        # https://numpy.org/doc/stable/reference/generated/numpy.histogram_bin_edges.html#numpy.histogram_bin_edges
        idm = np.argmax(count)
        mode = values[idm]
        modes.append(mode)

    modes = np.array(modes)  # Scale modes
    modes -= np.mean(modes)

    # Bar plot
    plt.bar(np.arange(1, 7), -modes, color=colors)
    # plt.title("Amount of information of each well")
    plt.title(f"{fig_name}")
    plt.xlabel("Well ID")
    plt.ylabel("Opposite deviation from mode's mean")
    plt.grid(color="#95a5a6", linestyle="-", linewidth=0.5, axis="y", alpha=0.7)

    legend_a = _proxy_annotate(annotation=[alphabet[an_i + 1]], loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    plt.savefig(
        os.path.join(directory, f"{fig_name}_well_mode.png"),
        dpi=300,
        transparent=False,
    )
    plt.close()

    # Plot BOX
    columns = ["1", "2", "3", "4", "5", "6"]
    wmd = pd.DataFrame(columns=columns, data=wm.T)
    palette = {columns[i]: colors[i] for i in range(len(columns))}
    # palette = {'b', 'g', 'r', 'c', 'm', 'y'}
    fig, ax1 = plt.subplots()
    sns.boxplot(data=wmd, palette=palette, order=columns, linewidth=1, ax=ax1)
    [line.set_color("white") for line in ax1.get_lines()[4::6]]
    plt.ylim([-2.5, 3])
    plt.xlabel("Well ID")
    plt.ylabel("Metric value")
    if title is None:
        title = "Box-plot of the metric values for each data source"
    plt.title(title)
    plt.grid(color="saddlebrown", linestyle="--", linewidth=0.7, axis="y", alpha=0.5)

    try:
        an_i = int(directory.split("split")[-1])
    except ValueError:
        pass
    legend_a = _proxy_annotate(annotation=[alphabet[an_i]], loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    legend_b = _proxy_annotate(annotation=[f"Fold {an_i + 1}"], loc=1, fz=14)
    plt.gca().add_artist(legend_b)

    plt.savefig(
        os.path.join(directory, f"{fig_name}_well_box.png"),
        dpi=300,
        transparent=False,
    )
    plt.close()

    # Plot histogram
    for i, m in enumerate(wm):
        sns.kdeplot(m, color=f"{colors[i]}", shade=True, linewidth=2)
    # plt.title("Summed metric distribution for each well")
    plt.title(f"{fig_name}")
    plt.xlabel("Summed metric")
    plt.ylabel("KDE")
    legend_1 = plt.legend(wid, loc=2)
    plt.gca().add_artist(legend_1)
    plt.grid(alpha=0.2)

    legend_a = _proxy_annotate(annotation=[alphabet[an_i]], loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    plt.savefig(
        os.path.join(directory, f"{fig_name}_hist.png"),
        dpi=300,
        transparent=False,
    )
    plt.close()


def curves(
    cols: list,
    tc: np.array,
    highlight: list = None,
    ghost: bool = False,
    sdir: str = None,
    labelsize: float = 12,
    factor: float = 1,
    conc: bool = 0,
    xlabel: str = None,
    ylabel: str = None,
    title: str = "curves",
    show: bool = False,
):
    """
    Shows every breakthrough curve stacked on a plot.
    :param cols: List of colors
    :param ylabel:
    :param xlabel:
    :param factor:
    :param labelsize:
    :param tc: Curves with shape (n_sim, n_wells, n_time_steps)
    :param highlight: list: List of indices of curves to highlight in the plot
    :param ghost: bool: Flag to only display highlighted curves.
    :param sdir: Directory in which to save figure
    :param title: str: Title
    :param show: Whether to show or not
    """
    if highlight is None:
        highlight = []

    if not conc:
        n_sim, n_wells, nts = tc.shape
        for i in range(n_sim):
            for t in range(n_wells):
                if i in highlight:
                    plt.plot(tc[i][t] * factor, color=cols[t], linewidth=2, alpha=1)
                elif not ghost:
                    plt.plot(tc[i][t] * factor, color=cols[t], linewidth=0.2, alpha=0.5)
    else:
        plt.plot(np.concatenate(tc[0]) * factor, color=cols[0], linewidth=2, alpha=1)

    plt.grid(linewidth=0.3, alpha=0.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tick_params(labelsize=labelsize)
    if sdir:
        utils.dirmaker(sdir)
        plt.savefig(jp(sdir, f"{title}.png"), dpi=300, transparent=False)
        plt.close()
    if show:
        plt.show()
        plt.close()


def curves_i(
    cols: list,
    tc: np.array,
    highlight: list = None,
    labelsize: float = 12,
    factor: float = 1,
    xlabel: str = None,
    ylabel: str = None,
    sdir: str = None,
    show: bool = False,
):
    """
    Shows every breakthrough individually for each observation point.
    Will produce n_well figures of n_sim curves each.
    :param cols: List of colors
    :param labelsize:
    :param factor:
    :param xlabel:
    :param ylabel:
    :param tc: Curves with shape (n_sim, n_wells, n_time_steps)
    :param highlight: list: List of indices of curves to highlight in the plot
    :param sdir: Directory in which to save figure
    :param show: Whether to show or not

    """
    if highlight is None:
        highlight = []
    title = "curves"
    n_sim, n_wels, nts = tc.shape
    for t in range(n_wels):
        for i in range(n_sim):
            if i in highlight:
                plt.plot(tc[i][t] * factor, color="k", linewidth=2, alpha=1)
            else:
                plt.plot(tc[i][t] * factor, color=cols[t], linewidth=0.2, alpha=0.5)
        colors = [cols[t], "k"]
        plt.grid(linewidth=0.3, alpha=0.4)
        plt.tick_params(labelsize=labelsize)
        # plt.title(f'Well {t + 1}')

        alphabet = string.ascii_uppercase
        legend_a = _proxy_annotate([f"{alphabet[t]}. Well {t + 1}"], fz=12, loc=2)

        labels = ["Training", "Test"]
        _proxy_legend(legend1=legend_a, colors=colors, labels=labels, loc=1)

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        if sdir:
            utils.dirmaker(sdir)
            plt.savefig(jp(sdir, f"{title}_{t + 1}.png"), dpi=300, transparent=False)
            plt.close()
        if show:
            plt.show()
            plt.close()


def plot_wells(wells: Setup.Wells, well_ids: list = None, markersize: float = 4.0):
    if well_ids is None:
        comb = [0] + list(wells.combination)
    else:
        comb = well_ids
    # comb = [0] + list(self.wells.combination)
    # comb = [0] + list(self.wells.combination)
    keys = [list(wells.wells_data.keys())[i] for i in comb]
    wbd = {k: wells.wells_data[k] for k in keys if k in wells.wells_data}
    s = 0
    for i in wbd:
        n = comb[s]
        if n == 0:
            label = "pw"
        else:
            label = f"{n}"
        if n in comb:
            plt.plot(
                wbd[i]["coordinates"][0],
                wbd[i]["coordinates"][1],
                f'{wbd[i]["color"]}o',
                markersize=markersize,
                markeredgecolor="k",
                markeredgewidth=0.5,
                label=label,
            )
        s += 1


def plot_pc_ba(
    bel,
    base_dir: str = None,
    root: str = None,
    w: str = None,
    data: bool = False,
    target: bool = False,
):
    """
    Comparison between original variables and the same variables back-transformed with n PCA components.
    :param w:
    :param base_dir:
    :param bel:
    :param root:
    :param data:
    :param target:
    :return:
    """

    if isinstance(root, (list, tuple)):
        if len(root) > 1:
            logger.error("Input error")
            return
        else:
            root = root[0]

    subdir = os.path.join(base_dir, root)

    if data:
        # Plot parameters for predictor
        xlabel = "Observation index number"
        ylabel = "Concentration ($g/m^{3})$"
        factor = 1000
        labelsize = 11
        d_pca_inverse_plot(
            bel,
            root=root,
            xlabel=xlabel,
            ylabel=ylabel,
            labelsize=labelsize,
            factor=factor,
            fig_dir=os.path.join(subdir, w, "pca"),
        )
    if target:
        h_pca_inverse_plot(bel, fig_dir=os.path.join(subdir, w, "pca"))


def plot_whpa(bel, base_dir):
    """
    Loads target pickle and plots all training WHPA
    :return:
    """

    h_training = bel.Y.reshape(bel.Y_shape)

    whpa_plot(
        whpa=h_training, highlight=True, halpha=0.5, lw=0.1, color="darkblue", alpha=0.5
    )

    h_pred = bel.Y_obs.reshape(bel.Y_shape)
    whpa_plot(
        whpa=h_pred,
        color="darkred",
        lw=1,
        alpha=1,
        annotation=[],
        xlabel="X(m)",
        ylabel="Y(m)",
        labelsize=11,
    )

    labels = ["Training", "Test"]
    legend = _proxy_annotate(annotation=[], loc=2, fz=14)
    _proxy_legend(
        legend1=legend,
        colors=["darkblue", "darkred"],
        labels=labels,
        fig_file=os.path.join(base_dir, "whpa_training.png"),
    )


def pca_vision(
    bel,
    base_dir: str,
    root: str or list = None,
    w: str = None,
    d: bool = True,
    h: bool = True,
    scores: bool = True,
    exvar: bool = True,
    before_after: bool = True,
    labels: bool = True,
):
    """
    Loads PCA pickles and plot scores for all folders
    :param before_after:
    :param base_dir:
    :param bel: BEL model
    :param w:
    :param labels:
    :param root: str:
    :param d: bool:
    :param h: bool:
    :param scores: bool:
    :param exvar: bool:
    :return:
    """

    if root is None:
        root = ""
    if w is None:
        w = ""

    subdir = jp(base_dir, root, w, "pca")

    if d:
        fig_file = os.path.join(subdir, "d_scores.png")
        if scores:
            pca_scores(
                training=bel.X_pc,
                prediction=bel.X_obs_pc,
                n_comp=bel.X_n_pc,
                # annotation=["C"],
                labels=labels,
                fig_file=fig_file,
            )
        # Explained variance plots
        if exvar:
            fig_file = os.path.join(subdir, "d_exvar.png")
            explained_variance(
                bel,
                n_comp=bel.X_n_pc,
                thr=0.8,
                # annotation=["E"],
                fig_file=fig_file,
            )
        if before_after:
            plot_pc_ba(bel, base_dir=base_dir, root=root, w=w, data=True, target=False)
    if h:
        # Transform and split
        h_pc_training = bel.Y_pc
        try:
            Y_obs = check_array(bel.Y_obs, allow_nd=True)
        except ValueError:
            Y_obs = check_array(bel.Y_obs.to_numpy().reshape(1, -1))
        h_pc_prediction = bel.Y_pre_processing.transform(Y_obs)
        # Plot
        fig_file = os.path.join(subdir, "h_pca_scores.png")
        if scores:
            pca_scores(
                training=h_pc_training,
                prediction=h_pc_prediction,
                n_comp=bel.Y_n_pc,
                # annotation=["D"],
                labels=labels,
                fig_file=fig_file,
            )
        # Explained variance plots
        if exvar:
            fig_file = os.path.join(subdir, "h_pca_exvar.png")
            explained_variance(
                bel,
                n_comp=bel.Y_n_pc,
                thr=0.8,
                # annotation=["F"],
                fig_file=fig_file,
            )
        if before_after:
            plot_pc_ba(bel, base_dir=base_dir, root=root, w=w, data=False, target=True)


def d_pca_inverse_plot(
    bel,
    root,
    factor: float = 1.0,
    xlabel: str = None,
    ylabel: str = None,
    labelsize: float = 11.0,
    fig_dir: str = None,
    show: bool = False,
):
    """
    Plot used to compare the reproduction of the original physical space after PCA transformation.
    :param bel: BEL model
    :param xlabel:
    :param ylabel:
    :param labelsize:
    :param factor:
    :param fig_dir: str:
    :param show: bool:
    :return:
    """

    shape = bel.X_shape
    v_pc = bel.X_obs_pc

    nc = bel.X_pre_processing["pca"].n_components_
    dummy = np.zeros((1, nc))
    dummy[:, : v_pc.shape[1]] = v_pc

    v_pred = bel.X_pre_processing.inverse_transform(dummy).reshape((-1,) + shape)
    to_plot = np.copy(bel.X_obs).reshape((-1,) + shape)

    cols = ["r" for _ in range(shape[1])]
    highlights = [i for i in range(shape[1])]
    curves(cols=cols, tc=to_plot, factor=factor, highlight=highlights, conc=True)

    cols = ["b" for _ in range(shape[1])]
    curves(cols=cols, tc=v_pred, factor=factor, highlight=highlights, conc=True)

    # Add title inside the box
    an = ["A"]
    legend_a = _proxy_annotate(annotation=an, loc=2, fz=14)
    _proxy_legend(
        legend1=legend_a,
        colors=["red", "blue"],
        labels=["Physical", "Back transformed"],
        marker=["-", "-"],
        loc=1,
    )
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.tick_params(labelsize=labelsize)

    # Increase y axis by a small percentage for annotation in upper left corner
    yrange = np.max(to_plot * factor) * 1.15
    plt.ylim([0, yrange])

    if fig_dir is not None:
        utils.dirmaker(fig_dir)
        plt.savefig(jp(fig_dir, f"{root}_d.png"), dpi=300, transparent=False)
        plt.close()
    if show:
        plt.show()
        plt.close()
