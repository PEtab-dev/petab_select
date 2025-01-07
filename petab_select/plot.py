"""Visualization routines for model selection.

Plotting methods generally take a :class:`PlotData` object as input, which
can be used to specify information used by multiple plotting methods.
"""

import copy
import warnings
from typing import Any

import matplotlib
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import matplotlib.ticker
import networkx as nx
import numpy as np
import upsetplot

from . import analyze
from .constants import Criterion
from .model import VIRTUAL_INITIAL_MODEL, ModelHash
from .models import Models

LABEL_FONTSIZE = 16
"""The font size of axis labels."""
TICK_LABEL_FONTSIZE = LABEL_FONTSIZE - 4
"""The font size of axis tick labels."""
DEFAULT_NODE_COLOR = "darkgrey"
"""The default color of nodes in graph plots."""


__all__ = [
    "bar_criterion_vs_models",
    "graph_history",
    "graph_iteration_layers",
    "line_best_by_iteration",
    "scatter_criterion_vs_n_estimated",
    "upset",
    "PlotData",
    "LABEL_FONTSIZE",
    "TICK_LABEL_FONTSIZE",
    "DEFAULT_NODE_COLOR",
]


class _IdentiDict(dict):
    """Dummy dictionary that returns keys as values."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __getitem__(self, key):
        return key


class PlotData:
    """Manage data used in plots.

    Attributes:
        models:
            The models.
        criterion:
            The criterion.
        labels:
            Model labels. Defaults to :attr:`petab_select.Model.model_label`
            or :attr:`petab_select.Model.model_id` or
            :attr:`petab_select.Model.hash`.
            Keys are model hashes, values are the labels.
        relative_criterion:
            Whether to display relative criterion values.
        parameter_labels:
            Labels for parameters. Keys are parameter IDs, values are labels.
            To use the ``parameterName`` column of your PEtab parameters table,
            provide ``petab_problem.parameter_df["parameterName"].to_dict()``.
        colors:
            Colors for each model. Keys are model hashes, values are
            matplotlib color specifiers (
            https://matplotlib.org/stable/users/explain/colors/colors.html ).
    """

    def __init__(
        self,
        models: Models,
        criterion: Criterion | None = None,
        relative_criterion: bool = False,
        labels: dict[ModelHash, str] | None = None,
        parameter_labels: dict[str, str] | None = None,
        colors: dict[str, str] = None,
    ):
        self.models = models
        self.criterion = criterion
        self.relative_criterion = relative_criterion
        self.parameter_labels = parameter_labels or _IdentiDict()
        self.colors = colors or {}

        labels = labels or {}
        self.labels = self.get_default_labels() | labels
        self.labels0 = copy.deepcopy(self.labels)

    def get_default_labels(self) -> dict[ModelHash, str]:
        """Get default model labels."""
        return (
            {
                model.hash: model.model_label or model.model_id or model.hash
                for model in self.models
            }
            | {
                model.predecessor_model_hash: model.predecessor_model_hash
                for model in self.models
            }
            | {VIRTUAL_INITIAL_MODEL.hash: "Virtual\nInitial\nModel"}
        )

    def augment_labels(
        self,
        criterion: bool = False,
        added_parameters: bool = False,
        removed_parameters: bool = False,
    ) -> None:
        """Add information to the plotted model labels.

        N.B.: this resets any previous augmentation.

        Args:
            criterion:
                Whether to include the criterion values.
            added_parameters:
                Whether to include the added parameters, compared to the predecessor model.
            removed_parameters:
                Whether to include the removed parameters, compared to the predecessor model.
        """
        if criterion:
            if not self.criterion:
                raise ValueError(
                    "Please construct the `PlotData` object with a specified "
                    "`criterion` first."
                )
            criterion_values = self.models.get_criterion(
                criterion=self.criterion,
                relative=self.relative_criterion,
                as_dict=True,
            )
        if added_parameters or removed_parameters:
            parameter_changes = analyze.get_parameter_changes(
                models=self.models,
                as_dict=True,
            )

        self.labels = copy.deepcopy(self.labels0)
        for model_hash, model_label in self.labels.items():
            if model_hash == VIRTUAL_INITIAL_MODEL.hash:
                continue

            criterion_label = None
            added_parameters_label = None
            removed_parameters_label = None

            if criterion and model_hash in self.models:
                criterion_label = f"{criterion_values[model_hash]:.2f}"

            if added_parameters:
                added_parameters_label = (
                    "+ {"
                    + ",".join(
                        self.parameter_labels[parameter_id]
                        for parameter_id in sorted(
                            parameter_changes[model_hash][0]
                        )
                    )
                    + "}"
                )

            if removed_parameters:
                removed_parameters_label = (
                    "- {"
                    + ",".join(
                        self.parameter_labels[parameter_id]
                        for parameter_id in sorted(
                            parameter_changes[model_hash][1]
                        )
                    )
                    + "}"
                )

            self.labels[model_hash] = "\n".join(
                str(label_part)
                for label_part in [
                    self.labels[model_hash],
                    criterion_label,
                    added_parameters_label,
                    removed_parameters_label,
                ]
                if label_part is not None
            )


def _set_axis_fontsizes(ax: plt.Axes) -> None:
    """Set the axes (tick) label fontsizes to the global variables.

    Customize via :const:`petab_select.plot.LABEL_FONTSIZE` and
    :const:`petab_select.plot.TICK_LABEL_FONTSIZE`.
    """
    ax.xaxis.label.set_size(LABEL_FONTSIZE)
    ax.yaxis.label.set_size(LABEL_FONTSIZE)
    ax.xaxis.set_tick_params(which="major", labelsize=TICK_LABEL_FONTSIZE)
    ax.yaxis.set_tick_params(which="major", labelsize=TICK_LABEL_FONTSIZE)


def upset(plot_data: PlotData) -> dict[str, matplotlib.axes.Axes | None]:
    """Plot an UpSet plot of estimated parameters and criterion.

    Args:
        plot_data:
            The plot data.

    Returns:
        The plot axes (see documentation from the
        `upsetplot <https://upsetplot.readthedocs.io/>`__ package).
    """
    # Get delta criterion values
    values = np.array(
        plot_data.models.get_criterion(
            criterion=plot_data.criterion,
            relative=plot_data.relative_criterion,
        )
    )

    # Sort by criterion value
    index = np.argsort(values)
    values = values[index]
    labels = [
        model.get_estimated_parameter_ids()
        for model in np.array(plot_data.models)[index]
    ]

    with warnings.catch_warnings():
        # TODO remove warnings context manager when fixed in upsetplot package
        warnings.simplefilter(action="ignore", category=FutureWarning)
        series = upsetplot.from_memberships(
            labels,
            data=values,
        )
        axes = upsetplot.plot(
            series, totals_plot_elements=0, with_lines=False, sort_by="input"
        )
    axes["intersections"].set_ylabel(r"$\Delta$" + plot_data.criterion)
    _set_axis_fontsizes(ax=axes["intersections"])
    _set_axis_fontsizes(ax=axes["matrix"])
    return axes


def line_best_by_iteration(
    plot_data: PlotData,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plot the improvement in criterion across iterations.

    Args:
        plot_data:
            The plot data.
        ax:
            The axis to use for plotting.

    Returns:
        The plot axes.
    """
    best_by_iteration = analyze.get_best_by_iteration(
        models=plot_data.models,
        criterion=plot_data.criterion,
    )

    # FIGURE
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    linewidth = 3

    iterations = sorted(best_by_iteration)
    best_models = Models(
        [best_by_iteration[iteration] for iteration in iterations]
    )
    iteration_labels = [
        str(iteration)
        + f"\n({plot_data.labels.get(model.hash, model.model_id)})"
        for iteration, model in zip(iterations, best_models, strict=True)
    ]

    criterion_values = best_models.get_criterion(
        criterion=plot_data.criterion, relative=plot_data.relative_criterion
    )

    ax.plot(
        iteration_labels,
        criterion_values,
        linewidth=linewidth,
        color=DEFAULT_NODE_COLOR,
        marker="x",
        markersize=10,
        markeredgewidth=2,
        markeredgecolor="red",
        # edgecolor='k'
    )

    ax.get_xticks()
    ax.set_xticks(list(range(len(criterion_values))))
    ax.set_xlabel("Iteration and model")
    ax.set_ylabel(
        (r"$\Delta$" if plot_data.relative_criterion else "")
        + plot_data.criterion
    )
    # could change to compared_model_ids, if all models are plotted
    _set_axis_fontsizes(ax=ax)
    ytl = ax.get_yticks()
    ax.set_ylim([min(ytl), max(ytl)])
    # removing top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def graph_history(
    plot_data: PlotData,
    draw_networkx_kwargs: dict[str, Any] = None,
    spring_layout_kwargs: dict[str, Any] = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plot all calibrated models in the model space, as a directed graph.

    Args:
        plot_data:
            The plot data.
        draw_networkx_kwargs:
            Forwarded to ``networkx.draw_networkx``.
        spring_layout_kwargs:
            Forwarded to ``networkx.spring_layout``.
        ax:
            The axis to use for plotting.


    Returns:
        The plot axes.
    """
    default_spring_layout_kwargs = {"k": 1, "iterations": 20}
    if spring_layout_kwargs is None:
        spring_layout_kwargs = default_spring_layout_kwargs

    criterion_values = plot_data.models.get_criterion(
        criterion=plot_data.criterion,
        relative=plot_data.relative_criterion,
        as_dict=True,
    )

    default_draw_networkx_kwargs = {
        "node_color": DEFAULT_NODE_COLOR,
        "arrowstyle": "-|>",
        "node_shape": "s",
        "node_size": 2500,
        "edgecolors": "k",
    }
    if draw_networkx_kwargs is None:
        draw_networkx_kwargs = default_draw_networkx_kwargs
    G = analyze.get_graph(models=plot_data.models)
    # Set colors
    if label_diff := set(plot_data.colors).difference(G.nodes):
        raise ValueError(
            "Colors were provided for the following model labels, but "
            f"these are not in the graph: `{label_diff}`."
        )

    node_colors = [
        plot_data.colors.get(
            model_hash, default_draw_networkx_kwargs["node_color"]
        )
        for model_hash in G.nodes
    ]
    draw_networkx_kwargs.update({"node_color": node_colors})
    nx.relabel_nodes(G, mapping=plot_data.labels, copy=False)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))

    pos = nx.spring_layout(G, **spring_layout_kwargs)
    nx.draw_networkx(G, pos, ax=ax, **draw_networkx_kwargs)

    return ax


def bar_criterion_vs_models(
    plot_data: PlotData,
    ax: plt.Axes = None,
    bar_kwargs: dict[str, Any] = None,
) -> plt.Axes:
    """Plot all calibrated models and their criterion value.

    Args:
        plot_data:
            The plot data.
        ax:
            The axis to use for plotting.
        bar_kwargs:
            Passed to the matplotlib `ax.bar` call.

    Returns:
        The plot axes.
    """
    if bar_kwargs is None:
        bar_kwargs = {}

    if ax is None:
        _, ax = plt.subplots()

    criterion_values = plot_data.models.get_criterion(
        criterion=plot_data.criterion,
        relative=plot_data.relative_criterion,
    )

    if label_diff := set(plot_data.colors).difference(plot_data.labels):
        raise ValueError(
            "Colors were provided for the following model labels, but "
            f"these are not in the graph: {label_diff}"
        )

    bar_kwargs["color"] = [
        plot_data.colors.get(model_label, DEFAULT_NODE_COLOR)
        for model_label in criterion_values
    ]

    labels = [
        plot_data.labels[model_hash] for model_hash in plot_data.models.hashes
    ]
    ax.bar(labels, criterion_values, **bar_kwargs)
    ax.set_xlabel("Model")
    ax.set_ylabel(
        (r"$\Delta$" if plot_data.relative_criterion else "")
        + plot_data.criterion
    )
    _set_axis_fontsizes(ax=ax)

    return ax


def scatter_criterion_vs_n_estimated(
    plot_data: PlotData,
    ax: plt.Axes = None,
    scatter_kwargs: dict[str, str] = None,
    max_jitter: float = 0.2,
) -> plt.Axes:
    """Plot criterion values against number of estimated parameters.

    Args:
        plot_data:
            The plot data.
        ax:
            The axis to use for plotting.
        scatter_kwargs:
            Forwarded to ``matplotlib.axes.Axes.scatter``.
        max_jitter:
            Add noise to distinguish models with the same number of parameters
            and similar criterion values. This is a positive value that is the
            maximal difference to the original value.

    Returns:
        The plot axes.
    """
    if scatter_kwargs is None:
        scatter_kwargs = {}

    if hash_diff := set(plot_data.colors).difference(plot_data.models.hashes):
        raise ValueError(
            "Colors were provided for the following model hashes, but "
            f"these are not in the graph: {hash_diff}"
        )
    scatter_kwargs["c"] = [
        plot_data.colors.get(model_hash, DEFAULT_NODE_COLOR)
        for model_hash in plot_data.models.hashes
    ]

    n_estimated = []
    for model in plot_data.models:
        n_estimated.append(len(model.get_estimated_parameter_ids()))

    criterion_values = plot_data.models.get_criterion(
        criterion=plot_data.criterion, relative=plot_data.relative_criterion
    )

    if max_jitter:
        n_estimated = np.array(n_estimated, dtype=float)
        rng = np.random.default_rng()
        n_estimated += rng.uniform(
            -max_jitter, max_jitter, size=n_estimated.size
        )

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(
        n_estimated,
        criterion_values,
        **scatter_kwargs,
    )

    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    ax.set_xlabel("Number of estimated parameters")
    ax.set_ylabel(
        (r"$\Delta$" if plot_data.relative_criterion else "")
        + plot_data.criterion,
    )
    _set_axis_fontsizes(ax=ax)

    return ax


def graph_iteration_layers(
    plot_data: PlotData,
    ax: plt.Axes = None,
    draw_networkx_kwargs: dict[str, Any] | None = None,
    # colors: Dict[str, str] = None,
    colorbar_mappable: matplotlib.cm.ScalarMappable = None,
    # use_tex: bool = True,
) -> plt.Axes:
    """Graph the models of each iteration of model selection.

    Args:
        plot_data:
            The plot data.
        ax:
            The axis to use for plotting.
        draw_networkx_kwargs:
            Passed to the ``networkx.draw_networkx`` call.
        colorbar_mappable:
            Customize the colors.
            See documentation for the `mappable` argument of
            ``matplotlib.pyplot.colorbar``.

    Returns:
        The plot axes.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 10))

    default_draw_networkx_kwargs = {
        "arrowstyle": "-|>",
        "node_shape": "s",
        "node_size": 250,
        "edgecolors": "k",
    }
    if draw_networkx_kwargs is None:
        draw_networkx_kwargs = default_draw_networkx_kwargs

    model_criterion_values = plot_data.models.get_criterion(
        criterion=plot_data.criterion,
        relative=plot_data.relative_criterion,
        as_dict=True,
    )

    G = analyze.get_graph(models=plot_data.models)

    # The ordering of models into iterations
    ordering = [
        [model.hash for model in iteration_models]
        for iteration_models in analyze.group_by_iteration(
            plot_data.models
        ).values()
    ]
    if VIRTUAL_INITIAL_MODEL.hash in G.nodes:
        ordering.insert(0, [VIRTUAL_INITIAL_MODEL.hash])

    if colorbar_mappable is None:
        norm = matplotlib.colors.Normalize(
            vmin=min(model_criterion_values.values()),
            vmax=max(model_criterion_values.values()),
        )
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "", ["green", "white"]
        )
        colorbar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = ax.get_figure().colorbar(colorbar_mappable, ax=ax)
    cbar.ax.set_title(r"$\Delta$" + plot_data.criterion)

    # Model node positions
    # FIXME change positions so that top edge of topmost node, and bottom edge
    # of bottommost nodes, are at 1 and 0, respectively
    X = [
        (0 if len(ordering) == 1 else 1 / (len(ordering) - 1)) * i
        for i in range(0, len(ordering))
    ]
    Y = [
        [
            (0 if len(layer) == 1 else 1 / (len(layer) - 1)) * j
            for j in range(0, len(layer))
        ]
        for layer in ordering
    ]

    pos = {
        plot_data.labels[model_hash]: (X[i], Y[i][j])
        for i, layer in enumerate(ordering)
        for j, model_hash in enumerate(layer)
    }

    node_colors = [
        (
            colorbar_mappable.to_rgba(model_criterion_values[model_hash])
            if model_hash in model_criterion_values
            else DEFAULT_NODE_COLOR
        )
        for model_hash in G.nodes
    ]

    # Apply custom labels
    nx.relabel_nodes(G, mapping=plot_data.labels, copy=False)

    nx.draw_networkx(
        G, pos, ax=ax, node_color=node_colors, **draw_networkx_kwargs
    )

    # Add `n=...` labels
    N = [len(y) for y in Y]
    for x, n in zip(X, N, strict=True):
        ax.annotate(
            f"n={n}",
            xy=(x, 1.1),
            fontsize=draw_networkx_kwargs.get("font_size", 20),
            ha="center",
        )

    # Set margins for the axes so that nodes aren't clipped
    ax.margins(0.15)
    ax.axis("off")
    return ax
