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

import skbel.utils
from skbel.algorithms import KDE, kde_params

__all__ = [
    "_my_alphabet",
    "_proxy_legend",
    "_proxy_annotate",
    "explained_variance",
    "pca_scores",
    "cca_vision",
    "_despine",
    "_cca_plot",
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
    """Add a second legend to a figure @ bottom right (loc=4)
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
    try:
        proxys = [
            plt.plot([], marker[i], color=c, markeredgecolor=pec[i])
            for i, c in enumerate(colors)
        ]
    except ValueError:
        # in case of 3D plot
        proxys = [
            plt.plot([], [], marker[i], color=c, markeredgecolor=pec[i])
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
    :param fz: Font size
    :param loc: Location (default: 1 = upper right corner, 2 = upper left corner)
    """
    if obj is None:
        obj = plt
    if annotation is None:
        annotation = []
    try:
        legend_a = obj.legend(
            plt.plot([], linestyle=None, color="w", alpha=0, markeredgecolor=None),
            annotation,
            handlelength=0,
            handletextpad=0,
            fancybox=True,
            loc=loc,
            fontsize=fz,
        )
    except TypeError:
        legend_a = obj.legend(
            plt.plot(0, 0, linestyle=None, color="w", alpha=0, markeredgecolor=None),
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
    n_components,
    evr,
    n_cut: int = 0,
    annotation: list = None,
    fig_file: str = None,
    show: bool = False,
    **kwargs,
):
    """PCA explained variance plot.

    :param n_components: Number of components
    :param evr: Explained variance ratio
    :param n_cut: Number of components to display
    :param annotation: List of annotation(s)
    :param fig_file: Path to figure file
    :param show: Show figure
    """
    if not n_cut:
        n_cut = n_components
    # parse kwargs
    if "cs" in kwargs:
        cs = kwargs["cs"]
    else:
        cs = None

    if cs:
        mcs = evr
    else:
        mcs = np.cumsum(evr[:n_cut])

    plt.grid(alpha=0.1)

    # bars for aesthetics
    plt.bar(
        np.arange(n_cut),
        mcs * 100,
        color="lightblue",
        alpha=0.5,
    )
    # line for aesthetics
    plt.plot(
        np.arange(n_cut),
        mcs * 100,
        "-o",
        linewidth=1,
        markersize=2,
        alpha=0.8,
    )
    # Axes labels
    plt.xlabel("PC number", fontsize=10)
    plt.ylabel("Cumulative explained variance (%)", fontsize=11)
    # Legend
    legend_a = _proxy_annotate(annotation=annotation, loc=2, fz=14)
    plt.gca().add_artist(legend_a)
    plt.tick_params(labelsize=8)
    # set x ticks as integers from 1 to n_components
    plt.xticks(np.arange(n_cut), np.arange(1, n_cut + 1))
    # locator_params(axis="x", nbins=10, integer=True)

    if fig_file:
        skbel.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(
            fig_file,
            dpi=300,
            pad_inches=0.01,
            bbox_inches="tight",
            transparent=False,
        )
        if show:
            plt.show()
        plt.close()
    if show:
        plt.show()
        plt.close()


def pca_scores(
    training: np.array,
    prediction: np.array = None,
    pc_post: np.array = None,
    random_pcs: np.array = None,
    n_comp: int = 0,
    annotation: list = None,
    title: str = None,
    xlabel: str = None,
    ylabel: str = None,
    fig_file: str = None,
    add_legend: bool = True,
    show: bool = False,
):
    """PCA scores plot, displays scores of observations above those of
    training.

    :param pc_post: PCA scores of the posterior
    :param training: Training scores
    :param prediction: Test scores
    :param pc_post: PCA scores of the posterior (Y)
    :param random_pcs: Random PCA scores
    :param n_comp: How many components to show
    :param annotation: List of annotation(s)
    :param title: Title of the plot
    :param xlabel: Label of the x axis
    :param ylabel: Label of the y axis
    :param fig_file: Path to figure file
    :param add_legend: Add legend
    :param show: Show figure
    """
    if not n_comp:
        n_comp = training.shape[1]
    # Scores plot
    if annotation is None:
        annotation = []
    # Grid
    # plt.grid(alpha=0.2)

    # Plot all training scores
    # We assume that we have a training set
    colors = ["blue"]
    labels = ["Training"]
    plt.plot(
        training.T[:n_comp],
        "ob",
        markersize=8,
        alpha=0.12,
        markeredgecolor=None,
        markeredgewidth=0.0,
    )

    if pc_post is not None:
        colors += ["lightgreen"]
        labels += ["Posterior"]
        plt.plot(
            pc_post.T[:n_comp],
            "o",
            markerfacecolor="lightgreen",
            markersize=5,
            markeredgecolor="k",
            markeredgewidth=0.05,
            alpha=0.15,
        )

    if random_pcs is not None:
        colors += ["gray"]
        labels += ["Random"]
        # they have to start at ncomp + 1
        try:
            cuit = pc_post.shape[1]
            plt.plot(
                np.arange(cuit, cuit + random_pcs.shape[1]),
                random_pcs.T,
                "o",
                markerfacecolor="gray",
                markersize=5,
                markeredgecolor=None,
                markeredgewidth=0.0,
                alpha=0.1,
            )
        except:
            pass

    # if pc_post AND random_pcs, draw a vertical line at the end of pc_post
    if pc_post is not None and random_pcs is not None:
        plt.axvline(x=pc_post.shape[1] - 0.5, color="k", linestyle="--", linewidth=0.5)

    if prediction is not None:
        colors += ["red"]
        labels += ["Test"]
        # For each sample used for prediction:
        # Select observation
        pc_obs = prediction.reshape(1, -1)
        # Create beautiful spline to follow prediction scores
        # xnew = np.linspace(1, n_comp, 200)  # New points for plotting curve
        # try:
        #     spl = make_interp_spline(
        #         np.arange(1, n_comp + 1), pc_obs.T[:n_comp], k=3
        #     )  # type: BSpline
        #     power_smooth = spl(xnew)
        #     # I forgot why I had to put '-1'
        #     plt.plot(xnew - 1, power_smooth, "red", linewidth=1.2, alpha=0.4)
        # except ValueError:
        #     pass

        plt.plot(
            pc_obs.T[:n_comp],  # Plot observations scores
            "ro",
            markersize=7,
            markeredgecolor="w",
            markeredgewidth=0.4,
            alpha=0.7,
        )

    if labels:
        plt.title(title)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
    # make ticks bold
    plt.tick_params(labelsize=8, width=2)
    # plt.tick_params(labelsize=7, which="major", direction="in")
    plt.xticks(np.arange(n_comp), np.arange(1, n_comp + 1))
    # make them bold

    plt.ylim(np.min(training.T[:n_comp]) * 1.5, np.max(training.T[:n_comp]))

    # locator_params(axis="x", nbins=10, integer=True)

    # Add legend
    # Add title inside the box
    legend_a = _proxy_annotate(annotation=annotation, loc=2, fz=14)

    if add_legend:
        _proxy_legend(
            legend1=legend_a,
            colors=colors,
            labels=labels,
            marker=["o"] * len(colors),
            loc=3,
        )

    if fig_file:
        skbel.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(
            fig_file,
            dpi=300,
            pad_inches=0.01,
            bbox_inches="tight",
            transparent=False,
        )
        if show:
            plt.show()
        plt.close()
    if show:
        plt.show()
        plt.close()


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
    :param offset: Absolute distance, in points, spines should be moved away from the axes (negative values move spines
        inward). A single value applies to all spines; a dict can be used to set offset values per side.
    :param trim: If True, limit spines to the smallest and largest major tick on each non-despined axis.
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


def _cca_plot(
    X_scores,
    Y_scores,
    X_obs: np.array = None,
    Y_obs: np.array = None,
    samples=None,
    sdir: str = None,
    show: bool = False,
    annotation_callback=None,
    mode=None,
):
    """Plot the Canonical Variate Pairs.

    :param mode: mode of inference
    :param X_scores: Canonical variates for X
    :param Y_scores: Canonical variates for Y
    :param X_obs: The X observations.
    :param Y_obs: The Y observations.
    :param samples: The samples to plot.
    :param sdir: The directory to save the plot to.
    :param show: Whether to show the plot.
    :param annotation_callback: A callback function to annotate the plot.
    """
    n_components = X_scores.shape[1]
    cca_coefficient = np.corrcoef(X_scores.T, Y_scores.T).diagonal(
        offset=n_components
    )  # Gets correlation coefficient
    vmax = 1
    for comp_n in range(n_components):
        # Get figure default parameters
        ax_joint, ax_marg_x, ax_marg_y, ax_cb = _get_defaults_kde_plot()

        marginal_eval_x = KDE()  # KDE for the marginal x
        marginal_eval_y = KDE()  # KDE for the marginal y
        marginal_eval_samples = KDE()  # KDE for the marginal samples

        # support is cached
        kde_x, sup_x = marginal_eval_x(X_scores.T[comp_n].reshape(1, -1))
        kde_y, sup_y = marginal_eval_y(Y_scores.T[comp_n].reshape(1, -1))

        if cca_coefficient[comp_n] < 0.999:
            # Plot h posterior given d
            if mode == "kde":
                density, support, bw = kde_params(
                    x=X_scores.T[comp_n], y=Y_scores.T[comp_n], gridsize=200
                )  # Get KDE parameters
                xx, yy = support

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

                # try:
                #     reg = kde_functions[obs_n][comp_n][
                #         "function"
                #     ]  # Get the regressor
                #     check_is_fitted(reg)
                #     reg_pts = reg.predict(X_scores.T[comp_n].reshape(-1, 1))
                #     ax_joint.plot(
                #         X_scores.T[comp_n], reg_pts, "r", linewidth=2, alpha=0.7
                #     )
                # except Exception:  # If no regressor
                #     pass
        # Vertical line
        ax_joint.axvline(
            x=X_obs.T[comp_n],
            color="blue",
            linewidth=1,
            alpha=0.5,
            label="$d^{c}_{True}$",
        )
        # Horizontal line
        try:
            ax_joint.axhline(
                y=Y_obs.T[comp_n],  # noqa
                color="red",
                linewidth=1,
                alpha=0.5,
                label="$h^{c}_{True}$",
            )
        except UnboundLocalError:  # If Y_obs is None
            pass
        # Scatter plot
        ax_joint.plot(
            X_scores.T[comp_n],
            Y_scores.T[comp_n],
            "bo",
            markersize=7,
            markeredgecolor="w",
            markeredgewidth=0.2,
            alpha=0.4,
        )
        ax_joint.plot(
            np.ones(samples.shape[1]) * X_obs.T[comp_n],  #####
            samples.T[comp_n],
            marker="o",
            markerfacecolor="lightgreen",
            markersize=7,
            markeredgecolor=None,
            markeredgewidth=0,
            alpha=0.2,
        )  # Samples
        # Point
        try:
            ax_joint.plot(
                X_obs.T[comp_n],
                Y_obs.T[comp_n],
                "ro",
                markersize=7,
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
            x=X_obs.T[comp_n], ymax=0.25, color="blue", linewidth=1, alpha=0.5
        )
        ax_marg_x.legend(loc=2, fontsize=10)

        # Marginal y plot
        #  - Line plot
        ax_marg_y.plot(kde_y, sup_y, color="black", linewidth=0.8, alpha=1)
        #  - Fill to axis
        # ax_marg_y.fill_betweenx(
        #     sup_y, 0, kde_y, alpha=0.5, color="darkred", label="$p(h^{c})$"
        # )
        ax_marg_y.fill_betweenx(
            sup_y, 0, kde_y, alpha=0.5, color="blue", label="$p(h^{c})$"
        )
        #  - Notch indicating true value
        try:
            ax_marg_y.axhline(
                y=Y_obs.T[comp_n],
                xmax=0.25,
                color="red",
                linewidth=1,
                alpha=0.5,
            )
        except UnboundLocalError:
            pass

        kde_samples, sup_samples = marginal_eval_samples(
            samples[:, :, comp_n].reshape(1, -1)
        )  # noqa
        ax_marg_y.plot(kde_samples, sup_samples, color="k", linewidth=0.5, alpha=1)
        #  - Fill to axis
        ax_marg_y.fill_betweenx(
            sup_samples,
            0,
            kde_samples,
            # color="mediumorchid",
            color="lightgreen",
            alpha=1,
            label="$p(h^{c}|d^{c}_{*})$",
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
        # rule to name the pair by 1st, 2nd, 3rd, ...
        if comp_n == 0:
            extn = "st"
        elif comp_n == 1:
            extn = "nd"
        elif comp_n == 2:
            extn = "rd"
        else:
            extn = "th"
        an = [
            f"{subtitle}. {comp_n + 1}{extn} pair  - "
            + r"$\rho$"
            + f" = {round(cca_coefficient[comp_n], 3)}"
        ]
        legend_a = _proxy_annotate(obj=ax_joint, annotation=an, loc=2, fz=12)

        _proxy_legend(
            obj=ax_joint,
            legend1=legend_a,
            colors=["blue", "red", "lightgreen", "blue", "red"],
            labels=[
                "$Training$",
                "$Test$",
                "$Samples$",
                "$d^{c}_{*}$",
                "$h^{c}_{True}$",
            ],
            marker=["o", "o", "o", "-", "-"],
            pec=["k", "k", "k", None, None],
            fz=10,
        )

        if sdir:
            skbel.utils.dirmaker(sdir, erase=False)
            plt.savefig(
                jp(sdir, f"cca_{comp_n}.png"),
                dpi=300,
                pad_inches=0.01,
                bbox_inches="tight",
                transparent=False,
            )
            if show:
                plt.show()
        if show:
            plt.show()
        plt.close()

    # return annotation_callback  # Return the iterator so that it can be used again


def cca_vision(
    X_scores=None,
    Y_scores=None,
    X_obs: np.array = None,
    Y_obs: np.array = None,
    samples=None,
    n_cut=None,
    cplot=True,
    annotation_call=None,
    fig_dir: str = None,
    show: bool = False,
):
    """Visualize the CCA results.

    :param X_scores: CCA scores for X
    :param Y_scores: CCA scores for Y
    :param X_obs: X observations
    :param Y_obs: Y observations
    :param samples: Samples from the model
    :param n_cut: Only show the first n_cut components
    :param cplot: Plot the CCA correlation coefficients
    :param annotation_call: Annotation callback
    :param fig_dir: Base directory path
    :param show: Show figure
    """
    if fig_dir is None:
        fig_dir = ""

    if annotation_call is None:
        annotation_call = _yield_alphabet()

    n_components = X_scores.shape[1]

    if n_cut:
        X_scores = X_scores[:, :n_cut]
        Y_scores = Y_scores[:, :n_cut]
        samples = samples[:, :, :n_cut]
        X_obs = X_obs[:, :n_cut]
        Y_obs = Y_obs[:, :n_cut]

    # KDE plots which consume a lot of time.
    _cca_plot(
        X_scores,
        Y_scores,
        X_obs=X_obs,
        Y_obs=Y_obs,
        samples=samples,
        sdir=fig_dir,
        show=show,
        annotation_callback=annotation_call,
    )

    # CCA coefficient plot
    if cplot:
        cca_coefficient = np.corrcoef(X_scores.T, Y_scores.T).diagonal(
            offset=n_components
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
        plt.xticks(
            np.arange(len(cca_coefficient)), np.arange(1, len(cca_coefficient) + 1)
        )
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
            dpi=300,
            bbox_inches="tight",
            pad_inches=0,
            transparent=False,
        )
    if show:
        plt.show()
    plt.close()
