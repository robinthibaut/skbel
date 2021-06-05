"""Some visualization utilities"""

#  Copyright (c) 2021. Robin Thibaut, Ghent University
import itertools
import os
import string
from os.path import join as jp

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import legend
from numpy import ma
from scipy.interpolate import BSpline, make_interp_spline

import skbel.utils
from sklearn.utils import check_array
from skbel.algorithms import KDE, kde_params, posterior_conditional

__all__ = [
    "_my_alphabet",
    "_proxy_legend",
    "_proxy_annotate",
    "explained_variance",
    "pca_scores",
    "cca_plot",
    "_despine",
    "_kde_cca",
]


def _my_alphabet(az: int):
    """
    Method used to make custom figure annotations.
    :param az: Index
    :return:
    """
    alphabet = string.ascii_uppercase
    extended_alphabet = ["".join(i) for i in list(itertools.permutations(alphabet, 2))]

    if az <= 25:
        sub = alphabet[az]
    else:
        j = az - 26
        sub = extended_alphabet[j]

    return sub


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
    :return:
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
    """
    Places annotation (or title) within the figure box
    :param annotation: Must be a list of labels even of it only contains one label. Savvy ?
    :param fz: Fontsize
    :param loc: Location (default: 1 = upper right corner, 2 = upper left corner)
    :return:
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
    bel,
    n_comp: int = 0,
    thr: float = 1.0,
    annotation: list = None,
    fig_file: str = None,
    show: bool = False,
):
    """
    PCA explained variance plot
    :param bel
    :param n_comp: Number of components to display
    :param thr: float: Threshold
    :param annotation: List of annotation(s)
    :param fig_file:
    :param show:
    :return:
    """
    plt.grid(alpha=0.1)
    if not n_comp:
        n_comp = bel.X_pre_processing["pca"].n_components_

    # Index where explained variance is below threshold:
    ny = len(
        np.where(
            np.cumsum(bel.X_pre_processing["pca"].explained_variance_ratio_) < thr
        )[0]
    )
    # Explained variance vector:
    cum = (
        np.cumsum(bel.X_pre_processing["pca"].explained_variance_ratio_[:n_comp]) * 100
    )
    # x-ticks
    plt.xticks(
        np.concatenate([np.array([0]), np.arange(4, n_comp, 5)]),
        np.concatenate([np.array([1]), np.arange(5, n_comp + 5, 5)]),
        fontsize=11,
    )
    # Tricky y-ticks
    yticks = np.append(cum[:ny], cum[-1])
    plt.yticks(yticks, fontsize=8.5)
    # bars for aesthetics
    plt.bar(
        np.arange(n_comp),
        np.cumsum(bel.X_pre_processing["pca"].explained_variance_ratio_[:n_comp]) * 100,
        color="m",
        alpha=0.1,
    )
    # line for aesthetics
    plt.plot(
        np.arange(n_comp),
        np.cumsum(bel.X_pre_processing["pca"].explained_variance_ratio_[:n_comp]) * 100,
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
        plt.close()
    if show:
        plt.show()
        plt.close()


def pca_scores(
    training: np.array,
    prediction: np.array,
    n_comp: int,
    annotation: list,
    fig_file: str = None,
    labels: bool = True,
    show: bool = False,
):
    """
    PCA scores plot, displays scores of observations above those of training.
    :param labels:
    :param training: Training scores
    :param prediction: Test scores
    :param n_comp: How many components to show
    :param annotation: List of annotation(s)
    :param fig_file:
    :param show:
    :return:
    """
    # Scores plot
    # Grid
    plt.grid(alpha=0.2)
    # Ticks
    plt.xticks(
        np.concatenate([np.array([0]), np.arange(4, n_comp, 5)]),
        np.concatenate([np.array([1]), np.arange(5, n_comp + 5, 5)]),
    )  # Plot all training scores
    plt.plot(training.T[:n_comp], "ob", markersize=3, alpha=0.1)
    # plt.plot(training.T[:ut], '+w', markersize=.5, alpha=0.2)

    # For each sample used for prediction:
    for sample_n in range(len(prediction)):
        # Select observation
        pc_obs = prediction[sample_n]
        # Create beautiful spline to follow prediction scores
        xnew = np.linspace(1, n_comp, 200)  # New points for plotting curve
        spl = make_interp_spline(
            np.arange(1, n_comp + 1), pc_obs.T[:n_comp], k=3
        )  # type: BSpline
        power_smooth = spl(xnew)
        # I forgot why I had to put '-1'
        plt.plot(xnew - 1, power_smooth, "red", linewidth=1.2, alpha=0.9)

        plt.plot(
            pc_obs.T[:n_comp],  # Plot observations scores
            "ro",
            markersize=3,
            markeredgecolor="k",
            markeredgewidth=0.4,
            alpha=0.8,
            label=str(sample_n),
        )

    if labels:
        plt.title("Principal Components of training and test dataset")
        plt.xlabel("PC number")
        plt.ylabel("PC")
    plt.tick_params(labelsize=11)
    # Add legend
    # Add title inside the box
    legend_a = _proxy_annotate(annotation=annotation, loc=2, fz=14)
    _proxy_legend(
        legend1=legend_a,
        colors=["blue", "red"],
        labels=["Training", "Test"],
        marker=["o", "o"],
    )

    if fig_file:
        skbel.utils.dirmaker(os.path.dirname(fig_file))
        plt.savefig(fig_file, dpi=300, transparent=False)
        plt.close()
    if show:
        plt.show()


def cca_plot(
    bel,
    d: np.array,
    h: np.array,
    d_pc_prediction: np.array,
    sdir: str = None,
    show: bool = False,
):
    """
    CCA plots.
    Receives d, h PC components to be predicted, transforms them in CCA space and adds it to the plots.
    :param bel
    :param d: d CCA scores
    :param h: h CCA scores
    :param d_pc_prediction: d test PC scores
    :param sdir: str:
    :param show: bool:
    :return:
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

    fig : matplotlib figure, optional
        Figure to despine all axes of, defaults to the current figure.
    ax : matplotlib axes, optional
        Specific axes object to despine. Ignored if fig is provided.
    top, right, left, bottom : boolean, optional
        If True, remove that spine.
    offset : int or dict, optional
        Absolute distance, in points, spines should be moved away
        from the axes (negative values move spines inward). A single value
        applies to all spines; a dict can be used to set offset values per
        side.
    trim : bool, optional
        If True, limit spines to the smallest and largest major tick
        on each non-despined axis.

    Returns
    -------
    None

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
    height = 6
    ratio = 6
    space = 0

    xlim = None
    ylim = None
    marginal_ticks = False

    # Set up the subplot grid
    f = plt.figure(figsize=(height, height))
    gs = plt.GridSpec(ratio + 1, ratio + 1)

    # ax_joint = f.add_subplot(gs[1:, :-1])
    # ax_marg_x = f.add_subplot(gs[0, :-1], sharex=ax_joint)
    # ax_marg_y = f.add_subplot(gs[1:, -1], sharey=ax_joint)

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
    sdir: str = None,
    show: bool = False,
    dist_plot: bool = False,
):
    # Find max kde value (absolutely not optimal)
    vmax = 0
    for comp_n in range(bel.cca.n_components):
        # Working with final product of BEL (not with raw cca scores)
        hp, sup = posterior_conditional(
            X=bel.X_f.T[comp_n], Y=bel.Y_f.T[comp_n], X_obs=bel.X_obs_f.T[comp_n]
        )

        # Plot h posterior given d
        density, _ = kde_params(x=bel.X_f.T[comp_n], y=bel.Y_f.T[comp_n])
        maxloc = np.max(density)
        if vmax < maxloc:
            vmax = maxloc

    cca_coefficient = np.corrcoef(bel.X_f.T, bel.Y_f.T).diagonal(
        offset=bel.cca.n_components
    )  # Gets correlation coefficient

    try:
        Y_obs = check_array(bel.Y_obs)
    except ValueError:
        Y_obs = check_array(bel.Y_obs.to_numpy().reshape(1, -1))

    # Transform Y obs
    bel.Y_obs_f = bel.transform(Y=Y_obs)

    # load prediction object
    post_test = bel.random_sample(n_posts=400)

    for comp_n in range(bel.cca.n_components):
        # Get figure default parameters
        ax_joint, ax_marg_x, ax_marg_y, ax_cb = _get_defaults_kde_plot()

        # Conditional:
        hp, sup = posterior_conditional(
            X=bel.X_f.T[comp_n], Y=bel.Y_f.T[comp_n], X_obs=bel.X_obs_f.T[comp_n]
        )

        # Plot h posterior given d
        density, support = kde_params(
            x=bel.X_f.T[comp_n], y=bel.Y_f.T[comp_n], gridsize=200
        )
        xx, yy = support

        marginal_eval_x = KDE()
        marginal_eval_y = KDE()

        # support is cached
        kde_x, sup_x = marginal_eval_x(bel.X_f.T[comp_n])
        kde_y, sup_y = marginal_eval_y(bel.Y_f.T[comp_n])

        y_samp = post_test.T[comp_n]
        # use the same support as y
        kde_y_samp, sup_samp = marginal_eval_y(y_samp)

        # Filled contour plot
        # Mask values under threshold
        z = ma.masked_where(density <= np.finfo(np.float16).eps, density)
        # Filled contour plot
        # 'BuPu_r' is nice
        cf = ax_joint.contourf(
            xx, yy, z, cmap="coolwarm", levels=100, vmin=0, vmax=vmax
        )
        cb = plt.colorbar(cf, ax=[ax_cb], location="left")
        cb.ax.set_title("$KDE_{Gaussian}$", fontsize=10)
        # Vertical line
        ax_joint.axvline(
            x=bel.X_obs_f.T[comp_n],
            color="red",
            linewidth=1,
            alpha=0.5,
            label="$d^{c}_{True}$",
        )
        # Horizontal line
        ax_joint.axhline(
            y=bel.Y_obs_f.T[comp_n],
            color="deepskyblue",
            linewidth=1,
            alpha=0.5,
            label="$h^{c}_{True}$",
        )
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
        # Point
        ax_joint.plot(
            bel.X_obs_f.T[comp_n],
            bel.Y_obs_f.T[comp_n],
            "wo",
            markersize=5,
            markeredgecolor="k",
            alpha=1,
        )
        # Marginal x plot
        #  - Line plot
        ax_marg_x.plot(sup_x, kde_x, color="black", linewidth=0.5, alpha=1)
        #  - Fill to axis
        ax_marg_x.fill_between(
            sup_x, 0, kde_x, color="royalblue", alpha=0.5, label="$p(d^{c})$"
        )
        #  - Notch indicating true value
        ax_marg_x.axvline(
            x=bel.X_obs_f.T[comp_n], ymax=0.25, color="red", linewidth=1, alpha=0.5
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
        ax_marg_y.axhline(
            y=bel.Y_obs_f.T[comp_n],
            xmax=0.25,
            color="deepskyblue",
            linewidth=1,
            alpha=0.5,
        )
        # Marginal y plot with BEL
        #  - Line plot
        ax_marg_y.plot(kde_y_samp, sup_samp, color="black", linewidth=0.5, alpha=0)
        #  - Fill to axis
        ax_marg_y.fill_betweenx(
            sup_samp,
            0,
            kde_y_samp,
            color="teal",
            alpha=0.5,
            label="$p(h^{c}|d^{c}_{*})$" + f"{bel.mode.upper()}",
        )
        # Conditional distribution
        #  - Line plot
        ax_marg_y.plot(hp, sup, color="red", alpha=0)
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
        plt.tick_params(labelsize=14)

        # Add custom artists
        subtitle = _my_alphabet(comp_n)
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
            skbel.utils.dirmaker(sdir)
            plt.savefig(
                jp(sdir, f"cca_kde_{comp_n}.png"),
                bbox_inches="tight",
                dpi=300,
                transparent=False,
            )
            plt.close()
        if show:
            plt.show()
            plt.close()

        def posterior_distribution():
            # prior
            plt.plot(sup_y, kde_y, color="black", linewidth=0.5, alpha=1)
            plt.fill_between(
                sup_y, 0, kde_y, color="mistyrose", alpha=1, label="$p(h^{c})$"
            )
            # posterior kde
            plt.plot(sup, hp, color="darkred", linewidth=0.5, alpha=0)
            plt.fill_between(
                sup,
                0,
                hp,
                color="salmon",
                alpha=0.5,
                label="$p(h^{c}|d^{c}_{*})$ (KDE)",
            )
            # posterior bel
            plt.plot(sup_samp, kde_y_samp, color="black", linewidth=0.5, alpha=1)
            plt.fill_between(
                sup_samp,
                0,
                kde_y_samp,
                color="gray",
                alpha=0.5,
                label="$p(h^{c}|d^{c}_{*})$ (BEL)",
            )

            # True prediction
            plt.axvline(
                x=bel.Y_obs_f[0],
                linewidth=3,
                alpha=0.4,
                color="deepskyblue",
                label="$h^{c}_{True}$",
            )

            # Grid
            plt.grid(alpha=0.2)

            # Tuning
            plt.ylabel("Density", fontsize=14)
            plt.xlabel("$h^{c}$", fontsize=14)
            plt.xlim([np.min(bel.X_f.T[comp_n]), np.max(bel.X_f.T[comp_n])])
            plt.tick_params(labelsize=14)

            plt.legend(loc=2)

            if sdir:
                skbel.utils.dirmaker(sdir)
                plt.savefig(
                    jp(sdir, f"cca_prior_post_{comp_n}.png"),
                    bbox_inches="tight",
                    dpi=300,
                    transparent=False,
                )
                plt.close()
            if show:
                plt.show()
                plt.close()

        if dist_plot:
            posterior_distribution()
