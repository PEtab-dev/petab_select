"""Visualization routines for model selection with pyPESTO."""

from typing import Any

import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from more_itertools import one
from toposort import toposort

from .constants import VIRTUAL_INITIAL_MODEL, Criterion
from .model import Model

RELATIVE_LABEL_FONTSIZE = -2
NORMAL_NODE_COLOR = "darkgrey"


def default_label_maker(model: Model) -> str:
    """Create a model label, for plotting."""
    return model.model_hash[:4]


def get_model_hashes(models: list[Model]) -> dict[str, Model]:
    model_hashes = {model.get_hash(): model for model in models}
    return model_hashes


def get_selected_models(
    models: list[Model],
    criterion: Criterion,
):
    criterion_value0 = np.inf
    model0 = None
    model_hashes = get_model_hashes(models)
    for model in models:
        criterion_value = model.get_criterion(criterion)
        if criterion_value < criterion_value0:
            criterion_value0 = criterion_value
            model0 = model

    selected_models = [model0]
    while True:
        model0 = selected_models[-1]
        model1 = model_hashes.get(model0.predecessor_model_hash, None)
        if model1 is None:
            break
        selected_models.append(model1)

    return selected_models[::-1]


def get_relative_criterion_values(
    criterion_values: dict[str, float] | list[float],
) -> dict[str, float] | list[float]:
    values = criterion_values
    if isinstance(criterion_values, dict):
        values = criterion_values.values()

    value0 = np.inf
    for value in values:
        if value < value0:
            value0 = value

    if isinstance(criterion_values, dict):
        return {k: v - value0 for k, v in criterion_values.items()}
    return [v - value0 for v in criterion_values]


def line_selected(
    models: list[Model],
    criterion: Criterion,
    relative: bool = True,
    fz: int = 14,
    size: tuple[float, float] = (5, 4),
    labels: dict[str, str] = None,
    ax: plt.Axes = None,
) -> plt.Axes:
    """Plot criterion for calibrated models.

    Parameters
    ----------
    models:
        A list of all models. The selected models will be inferred from the
        best model and its chain of predecessor model hashes.
    criterion:
        The criterion by which models are selected.
    relative:
        If `True`, criterion values are plotted relative to the lowest
        criterion value. TODO is the lowest value, always the best? May not
        be for different criterion.
    fz:
        fontsize
    size:
        Figure size in inches.
    labels:
        A dictionary of model labels, where keys are model hashes, and
        values are model labels, for plotting. If a model label is not
        provided, it will be generated from its model ID.
    ax:
        The axis to use for plotting.

    Returns
    -------
    matplotlib.pyplot.Axes
        The plot axis.
    """
    models = get_selected_models(models=models, criterion=criterion)

    if labels is None:
        labels = {}

    # FIGURE
    if ax is None:
        _, ax = plt.subplots(figsize=size)
    linewidth = 3

    models = [model for model in models if model != VIRTUAL_INITIAL_MODEL]

    criterion_values = {
        labels.get(model.get_hash(), model.model_id): model.get_criterion(
            criterion
        )
        for model in models
    }
    if relative:
        criterion_values = get_relative_criterion_values(criterion_values)

    ax.plot(
        criterion_values.keys(),
        criterion_values.values(),
        linewidth=linewidth,
        color=NORMAL_NODE_COLOR,
        # edgecolor='k'
    )

    ax.get_xticks()
    ax.set_xticks(list(range(len(criterion_values))))
    ax.set_ylabel(
        criterion + (" (relative)" if relative else " (absolute)"), fontsize=fz
    )
    # could change to compared_model_ids, if all models are plotted
    ax.set_xticklabels(
        criterion_values.keys(),
        fontsize=fz + RELATIVE_LABEL_FONTSIZE,
    )
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fz + RELATIVE_LABEL_FONTSIZE)
    ytl = ax.get_yticks()
    ax.set_ylim([min(ytl), max(ytl)])
    # removing top and right borders
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    return ax


def graph_history(
    models: list[Model],
    criterion: Criterion = None,
    ax: plt.Axes = None,
    labels: dict[str, str] = None,
    colors: dict[str, str] = None,
    optimal_distance: float = 1,
    options: dict = None,
    relative: bool = True,
) -> plt.Axes:
    """Plot all calibrated models in the model space, as a directed graph.

    TODO replace magic numbers with options/constants

    Parameters
    ----------
    models:
        A list of models.
    criterion:
        The criterion.
    optimal_distance:
        See docs for argument `k` in `networkx.spring_layout`.
    relative:
        If `True`, criterion values are offset by the minimum criterion
        value.
    options:
        Additional keyword arguments for `networkx.draw_networkx`.
    labels:
        A dictionary of model labels, where keys are model hashes, and
        values are model labels, for plotting. If a model label is not
        provided, it will be generated from its model ID.
    colors:
        Colors for each model, using their labels.
    ax:
        The axis to use for plotting.

    Returns
    -------
    matplotlib.pyplot.Axes
        The plot axis.
    """
    model_hashes = get_model_hashes(models)

    criterion_values = {
        model_hash: model.get_criterion(criterion)
        for model_hash, model in model_hashes.items()
    }
    if relative:
        criterion_values = get_relative_criterion_values(criterion_values)

    if labels is None:
        labels = {
            model_hash: model.model_id
            + (
                f"\n{criterion_values[model_hash]:.2f}"
                if criterion is not None
                else ""
            )
            for model_hash, model in model_hashes.items()
        }
    labels = labels.copy()
    labels[VIRTUAL_INITIAL_MODEL] = "Virtual\nInitial\nModel"

    G = nx.DiGraph()
    edges = []
    for model in models:
        predecessor_model_hash = model.predecessor_model_hash
        if predecessor_model_hash is not None:
            from_ = labels.get(predecessor_model_hash, predecessor_model_hash)
            # may only not be the case for
            # COMPARED_MODEL_ID == INITIAL_VIRTUAL_MODEL
            if predecessor_model_hash in model_hashes:
                predecessor_model = model_hashes[predecessor_model_hash]
                from_ = labels.get(
                    predecessor_model.get_hash(),
                    predecessor_model.model_id,
                )
        else:
            raise NotImplementedError(
                "Plots for models with `None` as their predecessor model are "
                "not yet implemented."
            )
            from_ = "None"
        to = labels.get(model.get_hash(), model.model_id)
        edges.append((from_, to))

    G.add_edges_from(edges)
    default_options = {
        "node_color": NORMAL_NODE_COLOR,
        "arrowstyle": "-|>",
        "node_shape": "s",
        "node_size": 2500,
    }
    if options is None:
        options = default_options
    if colors is not None:
        if label_diff := set(colors).difference(list(G)):
            raise ValueError(
                "Colors were provided for the following model labels, but "
                "these are not in the graph: {label_diff}"
            )

        node_colors = [
            colors.get(model_label, default_options["node_color"])
            for model_label in list(G)
        ]
        options.update({"node_color": node_colors})

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))

    pos = nx.spring_layout(G, k=optimal_distance, iterations=20)
    nx.draw_networkx(G, pos, ax=ax, **options)

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

    Parameters
    ----------
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

    Returns
    -------
    matplotlib.pyplot.Axes
        The plot axis.
    """
    model_hashes = get_model_hashes(models)

    if bar_kwargs is None:
        bar_kwargs = {}

    if labels is None:
        labels = {
            model_hash: model.model_id
            for model_hash, model in model_hashes.items()
        }

    if ax is None:
        _, ax = plt.subplots()

    criterion_values = {
        labels.get(model.get_hash(), model.model_id): model.get_criterion(
            criterion
        )
        for model in models
    }

    if colors is not None:
        if label_diff := set(colors).difference(criterion_values):
            raise ValueError(
                "Colors were provided for the following model labels, but "
                "these are not in the graph: {label_diff}"
            )

        bar_kwargs["color"] = [
            colors.get(model_label, NORMAL_NODE_COLOR)
            for model_label in criterion_values
        ]

    if relative:
        criterion_values = get_relative_criterion_values(criterion_values)
    ax.bar(criterion_values.keys(), criterion_values.values(), **bar_kwargs)
    ax.set_xlabel("Model labels")
    ax.set_ylabel(
        criterion.value + (" (relative)" if relative else " (absolute)")
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
) -> plt.Axes:
    """Plot criterion values against number of estimated parameters.

    Parameters
    ----------
    models:
        A list of models.
    criterion:
        The criterion.
    ax:
        The axis to use for plotting.
    relative:
        If `True`, criterion values are offset by the minimum criterion
        value.

    Returns
    -------
    matplotlib.pyplot.Axes
        The plot axis.
    """
    model_hashes = get_model_hashes(models)

    if labels is None:
        labels = {
            model_hash: model.model_id
            for model_hash, model in model_hashes.items()
        }

    if scatter_kwargs is None:
        scatter_kwargs = {}

    if colors is not None:
        if label_diff := set(colors).difference(labels.values()):
            raise ValueError(
                "Colors were provided for the following model labels, but "
                "these are not in the graph: {label_diff}"
            )
        scatter_kwargs["c"] = [
            colors.get(model_label, NORMAL_NODE_COLOR)
            for model_label in labels.values()
        ]

    n_estimated = []
    criterion_values = []
    for model in models:
        n_estimated.append(len(model.get_estimated_parameter_ids_all()))
        criterion_values.append(model.get_criterion(criterion))
    if relative:
        criterion_values = get_relative_criterion_values(criterion_values)

    if ax is None:
        _, ax = plt.subplots()

    ax.scatter(
        n_estimated,
        criterion_values,
        **scatter_kwargs,
    )

    ax.set_xlabel("Number of estimated parameters")
    ax.set_ylabel(
        criterion.value + (" (relative)" if relative else " (absolute)")
    )

    return ax


def graph_iteration_layers(
    models: list[Model],
    criterion: Criterion | None = None,
    labels: dict[str, str] = None,
    relative: bool = True,
    ax: plt.Axes = None,
    draw_networkx_kwargs: dict[str, Any] | None = None,
    # colors: Dict[str, str] = None,
    parameter_labels: dict[str, str] = None,
    augment_labels: bool = True,
    use_tex: bool = True,
) -> plt.Axes:
    """Graph the models of each iteration of model selection.

    Parameters
    ----------
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
    draw_networkx_kwargs:
        Passed to the `networkx.draw_networkx` call.
    parameter_labels:
        A dictionary of parameter labels, where keys are parameter IDs, and
        values are parameter labels, for plotting. Defaults to parameter IDs.
    augment_labels:
        If ``True'', provided labels will have estimated parameters and
        relative criterion values added to them, for plotting.

    Returns
    -------
    matplotlib.pyplot.Axes
        The plot axis.
    """
    if use_tex:
        rcParams0 = dict(plt.rcParams)
        rcParams = {
            "text.usetex": True,
            #'text.latex': {
            #    'preamble': r'\usepackage{color,xcolor}',
            # },
            "pgf.rcfonts": False,
            "pgf.preamble": r"\usepackage{color}",
        }
        plt.rcParams.update(rcParams)
        # for rcParam, rcParamValues in rcParams.items():
        #    plt.rc(rcParam, **rcParamValues)

    if ax is None:
        _, ax = plt.subplots(figsize=(20, 10))

    model_hashes = {model.get_hash(): model for model in models}

    default_draw_networkx_kwargs = {
        #'node_color': NORMAL_NODE_COLOR,
        "arrowstyle": "-|>",
        "node_shape": "s",
        "node_size": 250,
    }
    if draw_networkx_kwargs is None:
        draw_networkx_kwargs = default_draw_networkx_kwargs

    ancestry = {
        model.get_hash(): model.predecessor_model_hash for model in models
    }
    ancestry_as_set = {k: {v} for k, v in ancestry.items()}
    ordering = [list(hashes) for hashes in toposort(ancestry_as_set)]

    model_estimated_parameters = {
        model.get_hash(): set(model.estimated_parameters) for model in models
    }
    model_criterion_values = None
    if criterion is not None:
        model_criterion_values = {
            model.get_hash(): model.get_criterion(criterion)
            for model in models
        }

        min_criterion_value = min(model_criterion_values.values())
        model_criterion_values = {
            k: v - min_criterion_value
            for k, v in model_criterion_values.items()
        }

    model_parameter_diffs = {
        model.get_hash(): (
            (set(), set())
            if model.predecessor_model_hash not in model_estimated_parameters
            else (
                model_estimated_parameters[model.get_hash()].difference(
                    model_estimated_parameters[model.predecessor_model_hash]
                ),
                model_estimated_parameters[
                    model.predecessor_model_hash
                ].difference(model_estimated_parameters[model.get_hash()]),
            )
        )
        for model in models
    }

    if labels is None:
        labels = {model.get_hash(): model.model_id for model in models}
    if augment_labels:

        class Identidict(dict):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)

            def __getitem__(self, key):
                return key

        if parameter_labels is None:
            parameter_labels = Identidict()

        model_added_parameters = {
            model_hash: ";".join(
                [
                    parameter_labels[parameter_id]
                    for parameter_id in sorted(
                        model_parameter_diffs[model_hash][0]
                    )
                ]
            )
            for model_hash in model_estimated_parameters
        }
        model_removed_parameters = {
            model_hash: ";".join(
                [
                    parameter_labels[parameter_id]
                    for parameter_id in sorted(
                        model_parameter_diffs[model_hash][1]
                    )
                ]
            )
            for model_hash in model_estimated_parameters
        }

        labels = {
            model_hash: (
                label0
                # + (
                #    f'\n{model_criterion_values[model_hash]:.2f}'
                #    if model_hash in model_criterion_values
                #    else ''
                # )
                + (
                    (
                        # f'\n' +
                        r"\textcolor{green}{hi}"
                        # + ('\\textcolor{green}{' if use_tex else '') +
                        # f'{model_added_parameters[model_hash]}'
                        # + ('}' if use_tex else '')
                        if model_added_parameters.get(model_hash, "")
                        else ""
                    )
                    + (
                        # f'\n' +
                        r"\textcolor{red}{hi}"
                        # + ('\\textcolor{red}{' if use_tex else '') +
                        # f'{model_removed_parameters[model_hash]}'
                        # + ('}' if use_tex else '')
                        if model_removed_parameters.get(model_hash, "")
                        else ""
                    )
                )
            )
            for model_hash, label0 in labels.items()
        }

    norm = matplotlib.colors.Normalize(
        vmin=min(model_criterion_values.values()),
        vmax=max(model_criterion_values.values()),
    )
    cmap = matplotlib.cm.get_cmap("cool")
    scalar_mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    ax.get_figure().colorbar(scalar_mappable, ax=ax)

    G = nx.DiGraph(
        [
            (predecessor, successor)
            for successor, predecessor in ancestry.items()
        ]
    )
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

    # Labels
    # if labels is None:
    #    labels = {
    #        model.get_hash(): model.model_id
    #        + (
    #            f'\n{model.get_criterion(criterion):.2f}'
    #            if criterion is not None
    #            else ''
    #        )
    #        for model in models
    #    }
    labels[VIRTUAL_INITIAL_MODEL] = labels.get(
        VIRTUAL_INITIAL_MODEL, "Virtual\nInitial\nModel"
    )

    missing_labels = {
        model.get_hash(): model.model_id
        for model in models
        if model.get_hash() not in labels
    }
    missing_labels2 = {
        model.predecessor_model_hash: model.predecessor_model_hash
        for model in models
        if model.predecessor_model_hash not in labels
    }
    labels.update(missing_labels2)
    labels.update(missing_labels)
    # for label in missing_labels:
    #    labels[label] = label

    pos = {
        labels[model_hash]: (X[i], Y[i][j])
        for i, layer in enumerate(ordering)
        for j, model_hash in enumerate(layer)
    }
    nx.relabel_nodes(G, mapping=labels, copy=False)

    G_hashes = [
        one([k for k, v in labels.items() if v == label]) for label in G.nodes
    ]
    node_colors = [
        (
            scalar_mappable.to_rgba(model_criterion_values[model_hash])
            if model_hash in model_criterion_values
            else NORMAL_NODE_COLOR
        )
        for model_hash in G_hashes
    ]

    # if colors is not None:
    #    if label_diff := set(colors).difference(list(G)):
    #        raise ValueError(
    #            "Colors were provided for the following model labels, but "
    #            "these are not in the graph: {label_diff}"
    #        )

    #    node_colors = [
    #        colors.get(
    #            model_label,
    #            draw_networkx_kwargs.get(
    #                'node_color', default_draw_networkx_kwargs['node_color']
    #            ),
    #        )
    #        for model_label in list(G)
    #    ]
    #    draw_networkx_kwargs.update({'node_color': node_colors})
    nx.draw_networkx(
        G, pos, ax=ax, node_color=node_colors, **draw_networkx_kwargs
    )

    # Add `n=...` labels
    N = [len(y) for y in Y]
    for x, n in zip(X, N, strict=False):
        ax.annotate(
            f"n={n}",
            xy=(x, 1.1),
            fontsize=draw_networkx_kwargs.get("font_size", 20),
        )

    ## Get selected parameter IDs
    ## TODO move this logic elsewhere
    # selected_hashes = set(ancestry.values())
    # selected_models = {}
    # for model in models:
    #    if model.get_hash() in selected_hashes:
    #        selected_models[model.get_hash()] = model

    # selected_parameters = {
    #    model_hash: sorted(model.estimated_parameters)
    #    for model_hash, model in selected_models.items()
    # }

    # selected_order = [
    #    [model_hash for model_hash in layer if model_hash in selected_models]
    #    for layer in ordering
    # ]
    # selected_order = [
    #    None if not model_hash else one(model_hash)
    #    for model_hash in selected_order
    # ]

    # selected_parameter_ids = []
    # estimated0 = None
    # model_hash = None
    # for model_hash in selected_order:
    #    if model_hash is None:
    #        selected_parameter_ids.append('')
    #        continue
    #    if estimated0 is not None:
    #        new_parameter_ids = list(
    #            set(selected_parameters[model_hash]).symmetric_difference(
    #                estimated0
    #            )
    #        )
    #        new_parameter_names = []
    #        for new_parameter_id in new_parameter_ids:
    #            # Default to parameter ID, use parameter name if available
    #            new_parameter_name = new_parameter_id
    #            if (
    #                PARAMETER_NAME
    #                in selected_models[
    #                    model_hash
    #                ].petab_problem.parameter_df.columns
    #            ):
    #                petab_parameter_name = selected_models[
    #                    model_hash
    #                ].petab_problem.parameter_df.loc[
    #                    new_parameter_id, PARAMETER_NAME
    #                ]
    #                if not pd.isna(petab_parameter_name):
    #                    new_parameter_name = petab_parameter_name
    #            new_parameter_names.append(new_parameter_name)
    #        new_parameter_names = [
    #            new_parameter_name.replace('\\\\rightarrow ', '->')
    #            for new_parameter_name in new_parameter_names
    #        ]
    #        selected_parameter_ids.append(sorted(new_parameter_names))
    #    else:
    #        selected_parameter_ids.append([''])
    #    estimated0 = selected_parameters[model_hash]

    ## Add labels for selected parameters
    # for x, label in zip(X, selected_parameter_ids):
    #    ax.annotate(
    #        "\n".join(label),
    #        xy=(x, 1.15),
    #        fontsize=draw_networkx_kwargs.get('font_size', 20),
    #    )

    # Set margins for the axes so that nodes aren't clipped
    ax.margins(0.15)
    ax.axis("off")

    # FIXME
    plt.rcParams.update(rcParams0)
    return ax
