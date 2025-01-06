"""Visualization routines for model selection."""

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
from .model import VIRTUAL_INITIAL_MODEL, Model
from .models import Models

RELATIVE_LABEL_FONTSIZE = -2
NORMAL_NODE_COLOR = "darkgrey"


__all__ = [
    "bar_criterion_vs_models",
    "graph_history",
    "graph_iteration_layers",
    "line_best_by_iteration",
    "scatter_criterion_vs_n_estimated",
    "upset",
]


def upset(
    models: Models, criterion: Criterion
) -> dict[str, matplotlib.axes.Axes | None]:
    """Plot an UpSet plot of estimated parameters and criterion.

    Args:
        models:
            The models.
        criterion:
            The criterion.

    Returns:
        The plot axes (see documentation from the `upsetplot <https://upsetplot.readthedocs.io/>`__ package).
    """
    # Get delta criterion values
    values = np.array(models.get_criterion(criterion=criterion, relative=True))

    # Sort by criterion value
    index = np.argsort(values)
    values = values[index]
    labels = [
        model.get_estimated_parameter_ids()
        for model in np.array(models)[index]
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
    axes["intersections"].set_ylabel(r"$\Delta$" + criterion)
    return axes


def line_best_by_iteration(
    models: Models,
    criterion: Criterion,
    relative: bool = True,
    fz: int = 14,
    labels: dict[str, str] = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plot the improvement in criterion across iterations.

    Args:
        models:
            A list of all models. The selected models will be inferred from the
            best model and its chain of predecessor model hashes.
        criterion:
            The criterion by which models are selected.
        relative:
            If ``True``, criterion values are plotted relative to the lowest
            criterion value.
        fz:
            fontsize
        labels:
            A dictionary of model labels, where keys are model hashes, and
            values are model labels, for plotting. If a model label is not
            provided, it will be generated from its model ID.
        ax:
            The axis to use for plotting.

    Returns:
        The plot axes.
    """
    best_by_iteration = analyze.get_best_by_iteration(
        models=models, criterion=criterion
    )

    if labels is None:
        labels = {}

    # FIGURE
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    linewidth = 3

    iterations = sorted(best_by_iteration)
    best_models = Models(
        [best_by_iteration[iteration] for iteration in iterations]
    )
    iteration_labels = [
        str(iteration) + f"\n({labels.get(model.hash, model.model_id)})"
        for iteration, model in zip(iterations, best_models, strict=True)
    ]

    criterion_values = best_models.get_criterion(
        criterion=criterion, relative=relative
    )

    ax.plot(
        iteration_labels,
        criterion_values,
        linewidth=linewidth,
        color=NORMAL_NODE_COLOR,
        marker="x",
        markersize=10,
        markeredgewidth=2,
        markeredgecolor="red",
        # edgecolor='k'
    )

    ax.get_xticks()
    ax.set_xticks(list(range(len(criterion_values))))
    ax.set_xlabel("Iteration and model", fontsize=fz)
    ax.set_ylabel((r"$\Delta$" if relative else "") + criterion, fontsize=fz)
    # could change to compared_model_ids, if all models are plotted
    ax.set_xticklabels(
        ax.get_xticklabels(),
        fontsize=fz + RELATIVE_LABEL_FONTSIZE,
    )
    ax.yaxis.set_tick_params(
        which="major", labelsize=fz + RELATIVE_LABEL_FONTSIZE
    )
    ytl = ax.get_yticks()
    ax.set_ylim([min(ytl), max(ytl)])
    # removing top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def graph_history(
    models: Models,
    criterion: Criterion = None,
    labels: dict[str, str] = None,
    colors: dict[str, str] = None,
    draw_networkx_kwargs: dict[str, Any] = None,
    relative: bool = True,
    spring_layout_kwargs: dict[str, Any] = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plot all calibrated models in the model space, as a directed graph.

    Args:
        models:
            A list of models.
        criterion:
            The criterion.
        labels:
            A dictionary of model labels, where keys are model hashes, and
            values are model labels, for plotting. If a model label is not
            provided, it will be generated from its model ID.
        colors:
            Colors for each model, using their labels.
        draw_networkx_kwargs:
            Forwarded to ``networkx.draw_networkx``.
        relative:
            If ``True``, criterion values are offset by the minimum criterion
            value.
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

    criterion_values = models.get_criterion(
        criterion=criterion, relative=relative, as_dict=True
    )

    if labels is None:
        labels = {
            model.hash: model.model_id
            + (
                f"\n{criterion_values[model.hash]:.2f}"
                if criterion is not None
                else ""
            )
            for model in models
        }
    labels = labels.copy()
    labels[VIRTUAL_INITIAL_MODEL.hash] = "Virtual\nInitial\nModel"

    default_draw_networkx_kwargs = {
        "node_color": NORMAL_NODE_COLOR,
        "arrowstyle": "-|>",
        "node_shape": "s",
        "node_size": 2500,
        "edgecolors": "k",
    }
    if draw_networkx_kwargs is None:
        draw_networkx_kwargs = default_draw_networkx_kwargs
    G = analyze.get_graph(models=models, labels=labels)
    if colors is not None:
        if label_diff := set(colors).difference(list(G)):
            raise ValueError(
                "Colors were provided for the following model labels, but "
                f"these are not in the graph: {label_diff}"
            )

        node_colors = [
            colors.get(model_label, default_draw_networkx_kwargs["node_color"])
            for model_label in list(G)
        ]
        draw_networkx_kwargs.update({"node_color": node_colors})

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))

    pos = nx.spring_layout(G, **spring_layout_kwargs)
    nx.draw_networkx(G, pos, ax=ax, **draw_networkx_kwargs)

    return ax


def bar_criterion_vs_models(
    models: list[Model],
    criterion: Criterion = None,
    labels: dict[str, str] = None,
    relative: bool = True,
    ax: plt.Axes = None,
    bar_kwargs: dict[str, Any] = None,
    colors: dict[str, str] = None,
) -> plt.Axes:
    """Plot all calibrated models and their criterion value.

    Args:
        models:
            A list of models.
        criterion:
            The criterion.
        labels:
            A dictionary of model labels, where keys are model hashes, and
            values are model labels, for plotting. If a model label is not
            provided, it will be generated from its model ID.
        relative:
            If `True`, criterion values are offset by the minimum criterion
            value.
        ax:
            The axis to use for plotting.
        bar_kwargs:
            Passed to the matplotlib `ax.bar` call.
        colors:
            Custom colors. Keys are model hashes or ``labels`` (if provided),
            values are matplotlib colors.

    Returns:
        The plot axes.
    """
    if bar_kwargs is None:
        bar_kwargs = {}

    if labels is None:
        labels = {model.hash: model.model_id for model in models}

    if ax is None:
        _, ax = plt.subplots()

    bar_model_labels = [
        labels.get(model.hash, model.model_id) for model in models
    ]
    criterion_values = models.get_criterion(
        criterion=criterion, relative=relative
    )

    if colors is not None:
        if label_diff := set(colors).difference(bar_model_labels):
            raise ValueError(
                "Colors were provided for the following model labels, but "
                f"these are not in the graph: {label_diff}"
            )

        bar_kwargs["color"] = [
            colors.get(model_label, NORMAL_NODE_COLOR)
            for model_label in criterion_values
        ]

    ax.bar(bar_model_labels, criterion_values, **bar_kwargs)
    ax.set_xlabel("Model")
    ax.set_ylabel(
        (r"$\Delta$" if relative else "") + criterion,
    )

    return ax


def scatter_criterion_vs_n_estimated(
    models: list[Model],
    criterion: Criterion = None,
    relative: bool = True,
    ax: plt.Axes = None,
    colors: dict[str, str] = None,
    labels: dict[str, str] = None,
    scatter_kwargs: dict[str, str] = None,
    max_jitter: float = 0.2,
) -> plt.Axes:
    """Plot criterion values against number of estimated parameters.

    Args:
        models:
            A list of models.
        criterion:
            The criterion.
        relative:
            If `True`, criterion values are offset by the minimum criterion
            value.
        ax:
            The axis to use for plotting.
        colors:
            Custom colors. Keys are model hashes or ``labels`` (if provided),
            values are matplotlib colors.
        labels:
            A dictionary of model labels, where keys are model hashes, and
            values are model labels, for plotting. If a model label is not
            provided, it will be generated from its model ID.
        scatter_kwargs:
            Forwarded to ``matplotlib.axes.Axes.scatter``.
        max_jitter:
            Add noise to distinguish models with the same number of parameters
            and similar criterion values. This is a positive value that is the
            maximal difference to the original value.

    Returns:
        The plot axes.
    """
    labels = {
        model.hash: labels.get(model.model_id, model.model_id)
        for model in models
    }

    if scatter_kwargs is None:
        scatter_kwargs = {}

    if colors is not None:
        if label_diff := set(colors).difference(labels.values()):
            raise ValueError(
                "Colors were provided for the following model labels, but "
                f"these are not in the graph: {label_diff}"
            )
        scatter_kwargs["c"] = [
            colors.get(model_label, NORMAL_NODE_COLOR)
            for model_label in labels.values()
        ]

    n_estimated = []
    for model in models:
        n_estimated.append(len(model.get_estimated_parameter_ids()))

    criterion_values = models.get_criterion(
        criterion=criterion, relative=relative
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
        (r"$\Delta$" if relative else "") + criterion,
    )

    return ax


def graph_iteration_layers(
    models: list[Model],
    criterion: Criterion,
    labels: dict[str, str] = None,
    relative: bool = True,
    ax: plt.Axes = None,
    draw_networkx_kwargs: dict[str, Any] | None = None,
    # colors: Dict[str, str] = None,
    parameter_labels: dict[str, str] = None,
    augment_labels: bool = True,
    colorbar_mappable: matplotlib.cm.ScalarMappable = None,
    # use_tex: bool = True,
) -> plt.Axes:
    """Graph the models of each iteration of model selection.

    Args:
        models:
            A list of models.
        criterion:
            The criterion.
        labels:
            A dictionary of model labels, where keys are model hashes, and
            values are model labels, for plotting. If a model label is not
            provided, it will be generated from its model ID.
        relative:
            If ``True``, criterion values are offset by the minimum criterion
            value.
        ax:
            The axis to use for plotting.
        draw_networkx_kwargs:
            Passed to the ``networkx.draw_networkx`` call.
        parameter_labels:
            A dictionary of parameter labels, where keys are parameter IDs, and
            values are parameter labels, for plotting. Defaults to parameter IDs.
        augment_labels:
            If ``True``, provided labels will be augmented with the relative
            change in parameters.
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
        #'node_color': NORMAL_NODE_COLOR,
        "arrowstyle": "-|>",
        "node_shape": "s",
        "node_size": 250,
        "edgecolors": "k",
    }
    if draw_networkx_kwargs is None:
        draw_networkx_kwargs = default_draw_networkx_kwargs

    model_criterion_values = models.get_criterion(
        criterion=criterion, relative=relative, as_dict=True
    )

    parameter_changes = analyze.get_parameter_changes(
        models=models,
        as_dict=True,
    )

    G = analyze.get_graph(models=models)

    # The ordering of models into iterations
    ordering = [
        [model.hash for model in iteration_models]
        for iteration_models in analyze.group_by_iteration(models).values()
    ]
    if VIRTUAL_INITIAL_MODEL.hash in G.nodes:
        ordering.insert(0, [VIRTUAL_INITIAL_MODEL.hash])

    # Label customization
    labels = labels or {}
    labels[VIRTUAL_INITIAL_MODEL.hash] = labels.get(
        VIRTUAL_INITIAL_MODEL.hash, "Virtual\nInitial\nModel"
    )
    labels = (
        labels
        | {
            model.hash: model.model_id
            for model in models
            if model.hash not in labels
        }
        | {
            model.predecessor_model_hash: model.predecessor_model_hash
            for model in models
            if model.predecessor_model_hash not in labels
        }
    )
    if augment_labels:

        class Identidict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __getitem__(self, key):
                return key

        if parameter_labels is None:
            parameter_labels = Identidict()

        model_added_parameters = {
            model_hash: ",".join(
                [
                    parameter_labels[parameter_id]
                    for parameter_id in sorted(
                        parameter_changes[model_hash][0]
                    )
                ]
            )
            for model_hash in models.hashes
        }
        model_removed_parameters = {
            model_hash: ",".join(
                [
                    parameter_labels[parameter_id]
                    for parameter_id in sorted(
                        parameter_changes[model_hash][1]
                    )
                ]
            )
            for model_hash in models.hashes
        }

        labels = {
            model_hash: (
                label0
                if model_hash == VIRTUAL_INITIAL_MODEL.hash
                else "\n".join(
                    [
                        label0,
                        "+ {"
                        + model_added_parameters.get(model_hash, "")
                        + "}",
                        "- {"
                        + model_removed_parameters.get(model_hash, "")
                        + "}",
                    ]
                )
            )
            for model_hash, label0 in labels.items()
        }

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
    cbar.ax.set_title(r"$\Delta$" + criterion)

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
        labels[model_hash]: (X[i], Y[i][j])
        for i, layer in enumerate(ordering)
        for j, model_hash in enumerate(layer)
    }

    node_colors = [
        (
            colorbar_mappable.to_rgba(model_criterion_values[model_hash])
            if model_hash in model_criterion_values
            else NORMAL_NODE_COLOR
        )
        for model_hash in G.nodes
    ]

    # Apply custom labels
    nx.relabel_nodes(G, mapping=labels, copy=False)

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
