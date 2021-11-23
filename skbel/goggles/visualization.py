#  Copyright (c) 2021. Robin Thibaut, Ghent University

"""Some visualization utilities."""

import itertools
import os
import string
from os.path import join as jp

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import legend
from numpy import ma
from scipy.interpolate import BSpline, make_interp_spline
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted

import skbel.utils
from skbel.algorithms import KDE, kde_params, posterior_conditional

__all__ = [
    "_my_alphabet",
    "_proxy_legend",
    "_proxy_annotate",
    "explained_variance",
    "pca_scores",
    "pca_vision",
    "cca_vision",
    "cca_plot",
    "_despine",
    "_kde_cca",
]


def _my_alphabet(az: int):
    """Method used to make custom figure annotations.

    :param az: Index of the alphabet
    :return: corresponding letter
    """
    alphabet = string.ascii_uppercase
    extended_alphabet = ["".join(i) for i in list(itertools.permutations(alphabet, 2))]

    if az <= 25:
        sub = alphabet[az]
    else:
        j = az - 26
        sub = extended_alphabet[j]

    return sub


def _yield_alphabet(start=0):
    """Yields the alphabet from a given index.

    :param start: Index of the first letter
    """
    alphabet = string.ascii_uppercase
    extended_alphabet = ["".join(i) for i in list(itertools.permutations(alphabet, 2))]

    alphaomega = [char for char in alphabet] + extended_alphabet

    for sub in alphaomega[start:]:
        yield sub


def _proxy_legend(
    legend1: legend = None,
    colors: list = None,
    labels: list = None,
    loc: int = 4,
    marker: list = None,
    pec: list = None,
    fz: float = 11,
    fig_file: str = None,
    extra: list = None,
    obj=None,
):
    """
    Add a second legend to a figure @ bottom right (loc=4)
    https://stackoverflow.com/questions/12761806/matplotlib-2-different-legends-on-same-graph

    :param legend1: First legend instance from the figure
    :param colors: List of colors
    :param labels: List of labels
    :param loc: Position of the legend
    :param marker: Points 'o' or line '-'
    :param pec: List of point edge color, e.g. [None, 'k']
    :param fz: Fontsize
    :param fig_file: Path to figure file
    :param extra: List of extra elements to be added on the final figure
    """

    if obj is None:
        obj = plt
    # Default parameters
    if colors is None:
        colors = ["w"]
    if labels is None:
        labels = []
    if pec is None:
        pec = [None for _ in range(len(colors))]
    if extra is None:
        extra = []
    if marker is None:
        marker = ["-" for _ in range(len(colors))]

    # Proxy figures (empty plots)
    proxys = [
        plt.plot([], marker[i], color=c, markeredgecolor=pec[i])
        for i, c in enumerate(colors)
    ]

    leg = obj.legend([p[0] for p in proxys], labels, loc=loc, fontsize=fz)
    for text in leg.get_texts():
        text.set_color("k")

    if legend1:
        try:
            obj.gca().add_artist(legend1)
        except AttributeError:
            obj.add_artist(legend1)

    for el in extra:
        try:
            obj.gca().add_artist(el)
        except AttributeError:
            obj.add_artist(el)

    if fig_file:
        skbel.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, bbox_inches="tight", dpi=300, transparent=False)
        plt.close()


def _proxy_annotate(annotation: list = None, loc: int = 1, fz: float = 11, obj=None):
    """Places annotation (or title) within the figure box.

    :param annotation: Must be a list of labels even of it only contains one label. Savvy ?
    :param fz: Fontsize
    :param loc: Location (default: 1 = upper right corner, 2 = upper left corner)
    """
    if obj is None:
        obj = plt
    if annotation is None:
        annotation = []
    legend_a = obj.legend(
        plt.plot([], linestyle=None, color="w", alpha=0, markeredgecolor=None),
        annotation,
        handlelength=0,
        handletextpad=0,
        fancybox=True,
        loc=loc,
        fontsize=fz,
    )

    for text in legend_a.get_texts():
        text.set_color("k")

    return legend_a


def explained_variance(
    bel_pca,
    n_comp: int = 0,
    thr: float = 1.0,
    annotation: list = None,
    fig_file: str = None,
    show: bool = False,
):
    """PCA explained variance plot.

    :param bel_pca: PCA object
    :param n_comp: Number of components to display
    :param thr: float: Threshold
    :param annotation: List of annotation(s)
    :param fig_file: Path to figure file
    :param show: Show figure
    """
    plt.grid(alpha=0.1)
    if not n_comp:
        n_comp = bel_pca.n_components_

    # Index where explained variance is below threshold:
    ny = len(np.where(np.cumsum(bel_pca.explained_variance_ratio_) < thr)[0])
    # Explained variance vector:
    cum = np.cumsum(bel_pca.explained_variance_ratio_[:n_comp]) * 100
    # x-ticks
    try:
        plt.xticks(
            np.concatenate([np.array([0]), np.arange(4, n_comp, 5)]),
            np.concatenate([np.array([1]), np.arange(5, n_comp + 5, 5)]),
            fontsize=11,
        )
    except ValueError:
        plt.xticks(
            fontsize=11,
        )
    # Tricky y-ticks
    yticks = np.append(cum[:ny], cum[-1])
    plt.yticks(yticks, fontsize=8.5)
    # bars for aesthetics
    plt.bar(
        np.arange(n_comp),
        np.cumsum(bel_pca.explained_variance_ratio_[:n_comp]) * 100,
        color="m",
        alpha=0.1,
    )
    # line for aesthetics
    plt.plot(
        np.arange(n_comp),
        np.cumsum(bel_pca.explained_variance_ratio_[:n_comp]) * 100,
        "-o",
        linewidth=0.5,
        markersize=1.5,
        alpha=0.8,
    )
    # Axes labels
    plt.xlabel("PC number", fontsize=10)
    plt.ylabel("Cumulative explained variance (%)", fontsize=11)
    # Legend
    legend_a = _proxy_annotate(annotation=annotation, loc=2, fz=14)
    plt.gca().add_artist(legend_a)

    if fig_file:
        skbel.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, transparent=False)
        if show:
            plt.show()
        plt.close()
    if show:
        plt.show()
        plt.close()


def pca_scores(
    training: np.array,
    prediction: np.array = None,
    n_comp: int = None,
    annotation: list = None,
    fig_file: str = None,
    labels: bool = True,
    show: bool = False,
):
    """PCA scores plot, displays scores of observations above those of
    training.

    :param labels: labels for the plot
    :param training: Training scores
    :param prediction: Test scores
    :param n_comp: How many components to show
    :param annotation: List of annotation(s)
    :param fig_file: Path to figure file
    :param show: Show figure
    """
    # Scores plot
    if annotation is None:
        annotation = []
    # Grid
    plt.grid(alpha=0.2)

    # Plot all training scores
    plt.plot(training.T[:n_comp], "ob", markersize=3, alpha=0.1)

    if prediction is not None:
        # For each sample used for prediction:
        # Select observation
        pc_obs = prediction.reshape(1, -1)
        # Create beautiful spline to follow prediction scores
        xnew = np.linspace(1, n_comp, 200)  # New points for plotting curve
        try:
            spl = make_interp_spline(
                np.arange(1, n_comp + 1), pc_obs.T[:n_comp], k=3
            )  # type: BSpline
            power_smooth = spl(xnew)
            # I forgot why I had to put '-1'
            plt.plot(xnew - 1, power_smooth, "red", linewidth=1.2, alpha=0.9)
        except ValueError:
            pass

        plt.plot(
            pc_obs.T[:n_comp],  # Plot observations scores
            "ro",
            markersize=3,
            markeredgecolor="k",
            markeredgewidth=0.4,
            alpha=0.8,
            # label=str(sample_n),
        )

    if labels:
        plt.title("Principal Components")
        plt.xlabel("PC number")
        plt.ylabel("PC")
    plt.tick_params(labelsize=11)
    # Add legend
    # Add title inside the box
    legend_a = _proxy_annotate(annotation=annotation, loc=2, fz=14)

    if prediction is not None:
        _proxy_legend(
            legend1=legend_a,
            colors=["blue", "red"],
            labels=["Training", "Test"],
            marker=["o", "o"],
        )
    else:
        _proxy_legend(
            legend1=legend_a,
            colors=["blue"],
            labels=["Training"],
            marker=["o"],
        )

    if fig_file:
        skbel.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, transparent=False)
        if show:
            plt.show()
        plt.close()
    if show:
        plt.show()
        plt.close()


def cca_plot(
    bel,
    d: np.array,
    h: np.array,
    d_pc_prediction: np.array,
    sdir: str = None,
    show: bool = False,
):
    """CCA plots. Receives d, h PC components to be predicted, transforms them
    in CCA space and adds it to the plots.

    :param bel: BEL object
    :param d: d CCA scores
    :param h: h CCA scores
    :param d_pc_prediction: d test PC scores
    :param sdir: Path to save directory
    :param show: Show figure
    """

    cca_coefficient = np.corrcoef(d, h).diagonal(
        offset=bel.cca.n_components
    )  # Gets correlation coefficient

    # CCA plots for each observation:
    for i in range(bel.cca.n_components):
        for sample_n in range(len(d_pc_prediction)):  # For each 'observation'
            pass

        subtitle = _my_alphabet(i)

        # Add title inside the box
        an = [f"{subtitle}. Pair {i + 1} - R = {round(cca_coefficient[i], 3)}"]
        legend_a = _proxy_annotate(annotation=an, loc=2, fz=14)

        _proxy_legend(
            legend1=legend_a,
            colors=["black", "white"],
            labels=["Training", "Test"],
            marker=["o", "o"],
            pec=["k", "k"],
        )

        if sdir:
            skbel.utils.dirmaker(sdir)
            plt.savefig(
                jp(sdir, "cca{}.png".format(i)),
                bbox_inches="tight",
                dpi=300,
                transparent=False,
            )
            if show:
                plt.show()
            plt.close()
        if show:
            plt.show()
            plt.close()


def pca_vision(
    bel,
    d: np.array = None,
    h: np.array = None,
    obs_n: int = 0,
    X_obs: np.array = None,
    Y_obs: np.array = None,
    scores: bool = True,
    exvar: bool = True,
    thrx: float = 0.8,
    thry: float = 0.8,
    labels: bool = True,
    fig_dir: str = None,
    show: bool = False,
):
    """Loads PCA pickles and plot scores for all folders.

    :param bel: BEL object
    :param d: bool: Plot d scores
    :param h: bool: Plot h scores
    :param obs_n: Observation number
    :param X_obs: X_obs
    :param Y_obs: np.array: "True" target array
    :param scores: bool: Plot scores
    :param exvar: bool: Plot explained variance
    :param thrx: float: Threshold for X (explained variance plot)
    :param thry: float: Threshold for Y (explained variance plot)
    :param labels: Show labels
    :param fig_dir: Path to save directory
    :param show: Show figure
    """

    if fig_dir is None:
        fig_dir = ""

    if d is None:
        d = np.array([])

    if h is None:
        h = np.array([])

    annotation = _yield_alphabet()

    if d.any():
        X_pc = bel.X_pre_processing.transform(d)  # PCA scores
        if X_obs is not None:
            X_obs_pc = bel.X_pre_processing.transform(
                X_obs[obs_n]
            )  # PCA scores of the observed point
        else:
            X_obs_pc = None
        fig_file = os.path.join(fig_dir, "d_scores.png")
        if scores:
            pca_scores(
                training=X_pc,
                prediction=X_obs_pc,
                n_comp=X_pc.shape[1],
                annotation=[next(annotation)],
                labels=labels,
                fig_file=fig_file,
                show=show,
            )
        # Explained variance plots
        if exvar:
            fig_file = os.path.join(fig_dir, "d_exvar.png")
            try:
                explained_variance(
                    bel.X_pre_processing["pca"],
                    n_comp=X_pc.shape[1],
                    thr=thrx,
                    annotation=[next(annotation)],
                    fig_file=fig_file,
                    show=show,
                )
            except (AttributeError, KeyError):
                pass

    try:
        h_pc_training = bel.Y_pre_processing.transform(h)
        # Transform and split
        if Y_obs is not None:
            if type(Y_obs) is list:
                pass
            else:
                try:
                    Y_obs = check_array(Y_obs, allow_nd=True)
                except ValueError:
                    Y_obs = check_array(Y_obs.to_numpy().reshape(1, -1))
            h_pc_prediction = bel.Y_pre_processing.transform(Y_obs)
        else:
            h_pc_prediction = None
        # Plot
        fig_file = os.path.join(fig_dir, "h_pca_scores.png")
        if scores:
            pca_scores(
                training=h_pc_training,
                prediction=h_pc_prediction,
                n_comp=h_pc_training.shape[1],
                annotation=[next(annotation)],
                labels=labels,
                fig_file=fig_file,
                show=show,
            )
        # Explained variance plots
        if exvar:
            fig_file = os.path.join(fig_dir, "h_pca_exvar.png")
            try:
                explained_variance(
                    bel.Y_pre_processing["pca"],
                    n_comp=h_pc_training.shape[1],
                    thr=thry,
                    annotation=[next(annotation)],
                    fig_file=fig_file,
                    show=show,
                )
            except AttributeError:
                pass
    except Exception:
        pass


def _despine(
    fig=None,
    ax=None,
    top=True,
    right=True,
    left=False,
    bottom=False,
    offset=None,
    trim=False,
):
    """Remove the top and right spines from plot(s).

    :param fig: Figure to despine all axes of, defaults to the current figure.
    :param ax: Specific axes object to despine. Ignored if fig is provided.
    :param top, right, left, bottom: If True, remove that spine.
    :param offset: Absolute distance, in points, spines should be moved away
        from the axes (negative values move spines inward). A single value
        applies to all spines; a dict can be used to set offset values per
        side.
    :param trim: If True, limit spines to the smallest and largest major tick
        on each non-despined axis.
    """
    # Get references to the axes we want
    if fig is None and ax is None:
        axes_dummy = plt.gcf().axes
    elif fig is not None:
        axes_dummy = fig.axes
    elif ax is not None:
        axes_dummy = [ax]
    else:
        return

    for ax_i in axes_dummy:
        for side in ["top", "right", "left", "bottom"]:
            # Toggle the spine objects
            is_visible = not locals()[side]
            ax_i.spines[side].set_visible(is_visible)
            if offset is not None and is_visible:
                try:
                    val = offset.get(side, 0)
                except AttributeError:
                    val = offset
                ax_i.spines[side].set_position(("outward", val))

        # Potentially move the ticks
        if left and not right:
            maj_on = any(t.tick1line.get_visible() for t in ax_i.yaxis.majorTicks)
            min_on = any(t.tick1line.get_visible() for t in ax_i.yaxis.minorTicks)
            ax_i.yaxis.set_ticks_position("right")
            for t in ax_i.yaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.yaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if bottom and not top:
            maj_on = any(t.tick1line.get_visible() for t in ax_i.xaxis.majorTicks)
            min_on = any(t.tick1line.get_visible() for t in ax_i.xaxis.minorTicks)
            ax_i.xaxis.set_ticks_position("top")
            for t in ax_i.xaxis.majorTicks:
                t.tick2line.set_visible(maj_on)
            for t in ax_i.xaxis.minorTicks:
                t.tick2line.set_visible(min_on)

        if trim:
            # clip off the parts of the spines that extend past major ticks
            xticks = np.asarray(ax_i.get_xticks())
            if xticks.size:
                firsttick = np.compress(xticks >= min(ax_i.get_xlim()), xticks)[0]
                lasttick = np.compress(xticks <= max(ax_i.get_xlim()), xticks)[-1]
                ax_i.spines["bottom"].set_bounds(firsttick, lasttick)
                ax_i.spines["top"].set_bounds(firsttick, lasttick)
                newticks = xticks.compress(xticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_xticks(newticks)

            yticks = np.asarray(ax_i.get_yticks())
            if yticks.size:
                firsttick = np.compress(yticks >= min(ax_i.get_ylim()), yticks)[0]
                lasttick = np.compress(yticks <= max(ax_i.get_ylim()), yticks)[-1]
                ax_i.spines["left"].set_bounds(firsttick, lasttick)
                ax_i.spines["right"].set_bounds(firsttick, lasttick)
                newticks = yticks.compress(yticks <= lasttick)
                newticks = newticks.compress(newticks >= firsttick)
                ax_i.set_yticks(newticks)


def _get_defaults_kde_plot():
    """Get the default parameters for the kde plot."""
    height = 6
    ratio = 6
    space = 0

    xlim = None
    ylim = None
    marginal_ticks = False

    # Set up the subplot grid
    f = plt.figure(figsize=(height, height))
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    ax_joint = f.add_subplot(gs[1:, 1:-1])
    ax_marg_x = f.add_subplot(gs[0, 1:-1], sharex=ax_joint)
    ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)
    ax_cb = f.add_subplot(gs[1:, 0])

    fig = f
    ax_joint = ax_joint
    ax_marg_x = ax_marg_x
    ax_marg_y = ax_marg_y
    ax_cb = ax_cb

    # Turn off tick visibility for the measure axis on the marginal plots
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)

    plt.setp(ax_marg_x.get_xticklabels(minor=True), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(minor=True), visible=False)

    # Turn off the ticks on the density axis for the marginal plots
    plt.setp(ax_marg_x.yaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_x.yaxis.get_minorticklines(), visible=False)

    plt.setp(ax_marg_y.xaxis.get_majorticklines(), visible=False)
    plt.setp(ax_marg_y.xaxis.get_minorticklines(), visible=False)

    plt.setp(ax_marg_x.get_yticklabels(), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(), visible=False)

    plt.setp(ax_marg_x.get_yticklabels(minor=True), visible=False)
    plt.setp(ax_marg_y.get_xticklabels(minor=True), visible=False)

    ax_marg_x.yaxis.grid(False)
    ax_marg_y.xaxis.grid(False)

    ax_cb.axis("off")

    if xlim is not None:
        ax_joint.set_xlim(xlim)
    if ylim is not None:
        ax_joint.set_ylim(ylim)

    # Make the grid look nice
    _despine(f)
    if not marginal_ticks:
        _despine(ax=ax_marg_x, left=True)
        _despine(ax=ax_marg_y, bottom=True)

    for axes in [ax_marg_x, ax_marg_y, ax_cb]:
        for axis in [axes.xaxis, axes.yaxis]:
            axis.label.set_visible(False)
    f.tight_layout()
    f.subplots_adjust(hspace=space, wspace=space)

    return ax_joint, ax_marg_x, ax_marg_y, ax_cb


def _kde_cca(
    bel,
    obs_n: int = 0,
    X_obs: np.array = None,
    Y_obs: np.array = None,
    sdir: str = None,
    show: bool = False,
    annotation_callback=None,
):
    """Plot the kernel density estimate of the CCA.

    :param bel: The BEL object.
    :param obs_n: The index of the observation to plot.
    :param X_obs: The X observations.
    :param Y_obs: The Y observations.
    :param sdir: The directory to save the plot to.
    :param show: Whether to show the plot.
    :param annotation_callback: A callback function to annotate the plot.
    """
    cca_coefficient = np.corrcoef(bel.X_f.T, bel.Y_f.T).diagonal(
        offset=bel.cca.n_components
    )  # Gets correlation coefficient
    vmax = 1
    if type(X_obs) == list:
        pass
    else:
        try:
            X_obs = check_array(X_obs, allow_nd=True)
        except ValueError:
            try:
                X_obs = check_array(X_obs.to_numpy().reshape(1, -1))
            except AttributeError:
                X_obs = check_array(X_obs.reshape(1, -1))
    if type(Y_obs) == list:
        pass
    elif Y_obs is None:
        pass
    else:
        try:
            Y_obs = check_array(Y_obs, allow_nd=True)
        except ValueError:
            try:
                Y_obs = check_array(Y_obs.reshape(1, -1))
            except AttributeError:
                Y_obs = check_array(Y_obs.to_numpy().reshape(1, -1))

    # Transform X obs, Y obs
    try:
        X_obs_f, Y_obs_f = bel.transform(X=X_obs, Y=Y_obs)  # Transform X obs, Y obs
    except ValueError:
        X_obs_f = bel.transform(X=X_obs)

    # samples = bel.random_sample(
    #     X_obs_f=X_obs_f, obs_n=obs_n, n_posts=100
    # )  # Get 100 samples

    for comp_n in range(bel.cca.n_components):
        # Get figure default parameters
        ax_joint, ax_marg_x, ax_marg_y, ax_cb = _get_defaults_kde_plot()

        marginal_eval_x = KDE()  # KDE for the marginal x
        marginal_eval_y = KDE()  # KDE for the marginal y
        # support is cached
        kde_x, sup_x = marginal_eval_x(bel.X_f.T[comp_n].reshape(1, -1))
        kde_y, sup_y = marginal_eval_y(bel.Y_f.T[comp_n].reshape(1, -1))

        if cca_coefficient[comp_n] < 0.999:
            # Plot h posterior given d
            density, support, bw = kde_params(
                x=bel.X_f.T[comp_n],
                y=bel.Y_f.T[comp_n],
                gridsize=200,
            )  # Get KDE parameters
            xx, yy = support

            # Conditional:
            hp, sup = posterior_conditional(
                X_obs=X_obs_f.T[comp_n], dens=density, support=support, k=200
            )  # Get posterior

            # Filled contour plot
            # Mask values under threshold
            z = ma.masked_where(
                density <= np.finfo(np.float16).eps, density
            )  # Mask values under threshold
            # Filled contour plot
            # 'BuPu_r' is nice
            cf = ax_joint.contourf(
                xx, yy, z, cmap="coolwarm", levels=100, vmin=0, vmax=vmax
            )  # Filled contour plot
            cb = plt.colorbar(cf, ax=[ax_cb], location="left")  # Colorbar
            cb.ax.set_title("$KDE$", fontsize=10)  # Colorbar title

        try:
            reg = bel.kde_functions[obs_n][comp_n]["function"]  # Get the regressor
            check_is_fitted(reg)
            reg_pts = reg.predict(bel.X_f.T[comp_n].reshape(-1, 1))
            ax_joint.plot(bel.X_f.T[comp_n], reg_pts, "r", linewidth=2, alpha=0.7)
        except Exception:  # If no regressor
            pass
        # Vertical line
        ax_joint.axvline(
            x=X_obs_f.T[comp_n],
            color="red",
            linewidth=1,
            alpha=0.5,
            label="$d^{c}_{True}$",
        )
        # Horizontal line
        try:
            ax_joint.axhline(
                y=Y_obs_f.T[comp_n],  # noqa
                color="deepskyblue",
                linewidth=1,
                alpha=0.5,
                label="$h^{c}_{True}$",
            )
        except UnboundLocalError:  # If Y_obs is None
            pass
        # Scatter plot
        ax_joint.plot(
            bel.X_f.T[comp_n],
            bel.Y_f.T[comp_n],
            "ko",
            markersize=2,
            markeredgecolor="w",
            markeredgewidth=0.2,
            alpha=0.9,
        )
        # ax_joint.plot(
        #     np.ones(samples.shape[1]) * X_obs_f.T[comp_n],
        #     samples.T[comp_n],
        #     "go",
        #     alpha=0.3,
        # )  # Samples
        # Point
        try:
            ax_joint.plot(
                X_obs_f.T[comp_n],
                Y_obs_f.T[comp_n],
                "wo",
                markersize=5,
                markeredgecolor="k",
                alpha=1,
            )
        except UnboundLocalError:  # If Y_obs is None
            pass
        # Marginal x plot
        #  - Line plot
        ax_marg_x.plot(sup_x, kde_x, color="black", linewidth=0.5, alpha=1)
        #  - Fill to axis
        ax_marg_x.fill_between(
            sup_x, 0, kde_x, color="royalblue", alpha=0.5, label="$p(d^{c})$"
        )
        #  - Notch indicating true value
        ax_marg_x.axvline(
            x=X_obs_f.T[comp_n], ymax=0.25, color="red", linewidth=1, alpha=0.5
        )
        ax_marg_x.legend(loc=2, fontsize=10)

        # Marginal y plot
        #  - Line plot
        ax_marg_y.plot(kde_y, sup_y, color="black", linewidth=0.5, alpha=1)
        #  - Fill to axis
        ax_marg_y.fill_betweenx(
            sup_y, 0, kde_y, alpha=0.5, color="darkred", label="$p(h^{c})$"
        )
        #  - Notch indicating true value
        try:
            ax_marg_y.axhline(
                y=Y_obs_f.T[comp_n],
                xmax=0.25,
                color="deepskyblue",
                linewidth=1,
                alpha=0.5,
            )
        except UnboundLocalError:
            pass
        if cca_coefficient[comp_n] < 0.999:
            # Conditional distribution
            #  - Line plot
            ax_marg_y.plot(hp, sup, color="red", alpha=0)  # noqa
            #  - Fill to axis
            ax_marg_y.fill_betweenx(
                sup,
                0,
                hp,
                color="mediumorchid",
                alpha=0.5,
                label="$p(h^{c}|d^{c}_{*})_{KDE}$",
            )

        ax_marg_y.legend(fontsize=10)
        # Labels
        ax_joint.set_xlabel("$d^{c}$", fontsize=14)
        ax_joint.set_ylabel("$h^{c}$", fontsize=14)
        ax_joint.grid(alpha=0.5)
        plt.tick_params(labelsize=14)

        # Add custom artists
        subtitle = next(annotation_callback)
        # Add title inside the box
        an = [
            f"{subtitle}. Pair {comp_n + 1} - "
            + r"$\it{"
            + "r"
            + "}$"
            + f" = {round(cca_coefficient[comp_n], 3)}"
        ]
        legend_a = _proxy_annotate(obj=ax_joint, annotation=an, loc=2, fz=12)
        #
        _proxy_legend(
            obj=ax_joint,
            legend1=legend_a,
            colors=["black", "white", "red", "deepskyblue"],
            labels=["$Training$", "$Test$", "$d^{c}_{*}$", "$h^{c}_{True}$"],
            marker=["o", "o", "-", "-"],
            pec=["k", "k", None, None],
            fz=10,
        )

        if sdir:
            skbel.utils.dirmaker(sdir, erase=False)
            plt.savefig(
                jp(sdir, f"cca_kde_{comp_n}.png"),
                bbox_inches="tight",
                dpi=300,
                transparent=False,
            )
            if show:
                plt.show()
        if show:
            plt.show()
        plt.close()

    return annotation_callback  # Return the iterator so that it can be used again


def cca_vision(
    bel,
    X_obs: np.array,
    Y_obs: np.array,
    obs_n: int = None,
    fig_dir: str = None,
    show: bool = False,
):
    """Loads CCA pickles and plots components for all folders.

    :param bel: BEL model
    :param X_obs: Observed X (n_obs, n_comp)
    :param Y_obs: True target array
    :param obs_n: Observation number
    :param fig_dir: Base directory path
    :param show: Show figure
    """
    if fig_dir is None:
        fig_dir = ""

    annotation_call = _yield_alphabet()
    # KDE plots which consume a lot of time.
    _kde_cca(
        bel,
        X_obs=X_obs,
        Y_obs=Y_obs,
        obs_n=obs_n,
        sdir=fig_dir,
        show=show,
        annotation_callback=annotation_call,
    )

    # CCA coefficient plot
    cca_coefficient = np.corrcoef(bel.X_f.T, bel.Y_f.T).diagonal(
        offset=bel.cca.n_components
    )  # Gets correlation coefficient
    plt.plot(cca_coefficient, "lightblue", zorder=1)
    plt.scatter(
        x=np.arange(len(cca_coefficient)),
        y=cca_coefficient,
        c=cca_coefficient,
        alpha=1,
        s=50,
        cmap="coolwarm",
        zorder=2,
    )
    cb = plt.colorbar()
    cb.ax.set_title(r"$\it{" + "r" + "}$")
    plt.grid(alpha=0.4, linewidth=0.5, zorder=0)
    plt.xticks(np.arange(len(cca_coefficient)), np.arange(1, len(cca_coefficient) + 1))
    plt.tick_params(labelsize=5)
    plt.yticks([])
    # plt.title('Decrease of CCA correlation coefficient with component number')
    plt.ylabel("Correlation coefficient")
    plt.xlabel("Component number")

    # Add annotation
    legendary = _proxy_annotate(annotation=next(annotation_call), fz=14, loc=1)
    plt.gca().add_artist(legendary)

    plt.savefig(
        os.path.join(fig_dir, "coefs.png"),
        bbox_inches="tight",
        dpi=300,
        transparent=False,
    )
    if show:
        plt.show()
    plt.close()

    return annotation_call  # Return the iterator so that it can be used again
