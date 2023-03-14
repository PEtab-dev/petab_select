"""Visualization routines for model selection with pyPESTO."""
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import petab
from more_itertools import one
from toposort import toposort

from .constants import VIRTUAL_INITIAL_MODEL, Criterion
from .model import Model

RELATIVE_LABEL_FONTSIZE = -2


def default_label_maker(model: Model) -> str:
    """Create a model label, for plotting."""
    return model.model_hash[:4]


def get_model_hashes(models: List[Model]) -> Dict[str, Model]:
    model_hashes = {model.get_hash(): model for model in models}
    return model_hashes


def get_selected_models(
    models: List[Model],
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
    criterion_values: Union[Dict[str, float], List[float]],
) -> Union[Dict[str, float], List[float]]:
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
    models: List[Model],
    criterion: Criterion,
    relative: bool = True,
    fz: int = 14,
    size: Tuple[float, float] = (5, 4),
    labels: Dict[str, str] = None,
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
        color='lightgrey',
        # edgecolor='k'
    )

    ax.get_xticks()
    ax.set_xticks(list(range(len(criterion_values))))
    ax.set_ylabel(
        criterion + (' (relative)' if relative else ' (absolute)'), fontsize=fz
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
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    return ax


def graph_history(
    models: List[Model],
    criterion: Criterion = None,
    ax: plt.Axes = None,
    labels: Dict[str, str] = None,
    optimal_distance: float = 1,
    options: Dict = None,
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
                'Plots for models with `None` as their predecessor model are '
                'not yet implemented.'
            )
            from_ = 'None'
        to = labels.get(model.get_hash(), model.model_id)
        edges.append((from_, to))

    G.add_edges_from(edges)
    default_options = {
        'node_color': 'lightgrey',
        'arrowstyle': '-|>',
        'node_shape': 's',
        'node_size': 2500,
    }
    if options is not None:
        default_options.update(options)

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 12))

    pos = nx.spring_layout(G, k=optimal_distance, iterations=20)
    nx.draw_networkx(G, pos, ax=ax, **default_options)

    return ax


def bar_criterion_vs_models(
    models: List[Model],
    criterion: Criterion = None,
    labels: Dict[str, str] = None,
    relative: bool = True,
    ax: plt.Axes = None,
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

    if ax is None:
        _, ax = plt.subplots()

    criterion_values = {
        labels.get(model.get_hash(), model.model_id): model.get_criterion(
            criterion
        )
        for model in models
    }
    if relative:
        criterion_values = get_relative_criterion_values(criterion_values)
    ax.bar(criterion_values.keys(), criterion_values.values())
    ax.set_xlabel("Model labels")
    ax.set_ylabel(
        criterion.value + (' (relative)' if relative else ' (absolute)')
    )

    return ax


def scatter_criterion_vs_n_estimated(
    models: List[Model],
    criterion: Criterion = None,
    relative: bool = True,
    ax: plt.Axes = None,
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
    )

    ax.set_xlabel("Number of estimated parameters")
    ax.set_ylabel(
        criterion.value + (' (relative)' if relative else ' (absolute)')
    )

    return ax


def graph_iteration_layers(
    models: List[Model],
    criterion: Optional[Criterion] = None,
    labels: Dict[str, str] = None,
    relative: bool = True,
    ax: plt.Axes = None,
    draw_networkx_kwargs: Optional[Dict[str, Any]] = None,
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

    Returns
    -------
    matplotlib.pyplot.Axes
        The plot axis.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(20, 10))

    if labels is None:
        labels = {
            model.get_hash(): model.model_id
            + (
                f'\n{model.get_criterion(criterion):.2f}'
                if criterion is not None
                else ''
            )
            for model in models
        }
        labels[VIRTUAL_INITIAL_MODEL] = "Virtual\nInitial\nModel"
        missing_labels = [
            model.predecessor_model_hash
            for model in models
            if model.predecessor_model_hash not in labels
        ]
        for label in missing_labels:
            labels[label] = label

    if draw_networkx_kwargs is None:
        draw_networkx_kwargs = {
            'node_color': 'lightgrey',
            'arrowstyle': '-|>',
            'node_shape': 's',
            'node_size': 2500,
        }

    ancestry = {
        model.get_hash(): model.predecessor_model_hash for model in models
    }
    ancestry_as_set = {k: set([v]) for k, v in ancestry.items()}
    ordering = [list(hashes) for hashes in toposort(ancestry_as_set)]

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
    pos = {
        labels[model_hash]: (X[i], Y[i][j])
        for i, layer in enumerate(ordering)
        for j, model_hash in enumerate(layer)
    }
    nx.relabel_nodes(G, mapping=labels, copy=False)
    nx.draw_networkx(G, pos, ax=ax, **draw_networkx_kwargs)

    # Add `n=...` labels
    N = [len(y) for y in Y]
    for x, n in zip(X, N):
        ax.annotate(f'n={n}', xy=(x, 1.1), fontsize=12)

    # Get selected parameter IDs
    # TODO move this logic elsewhere
    selected_hashes = set(ancestry.values())
    selected_models = {}
    for model in models:
        if model.get_hash() in selected_hashes:
            selected_models[model.get_hash()] = model

    selected_parameters = {
        model_hash: sorted(model.estimated_parameters)
        for model_hash, model in selected_models.items()
    }

    selected_order = [
        [model_hash for model_hash in layer if model_hash in selected_models]
        for layer in ordering
    ]
    selected_order = [
        None if not model_hash else one(model_hash)
        for model_hash in selected_order
    ]

    selected_parameter_ids = []
    estimated0 = None
    model_hash = None
    for model_hash in selected_order:
        if model_hash is None:
            selected_parameter_ids.append('')
            continue
        if estimated0 is not None:
            new_parameter_ids = set(
                selected_parameters[model_hash]
            ).symmetric_difference(estimated0)
            new_parameter_names = [
                selected_models[model_hash].petab_problem.parameter_df.get(
                    'parameterName', new_parameter_id
                )
                for new_parameter_id in new_parameter_ids
            ]
            for index, new_parameter_name in enumerate(new_parameter_names):
                if not isinstance(new_parameter_name, str):
                    new_parameter_names[index] = new_parameter_name[
                        new_parameter_id
                    ]
            new_parameter_names = [
                new_parameter_name.replace('\\\\rightarrow ', '->')
                for new_parameter_name in new_parameter_names
            ]
            selected_parameter_ids.append(new_parameter_names)
        else:
            selected_parameter_ids.append([''])
        estimated0 = selected_parameters[model_hash]

    # Add labels for selected parameters
    for x, label in zip(X, selected_parameter_ids):
        ax.annotate("\n".join(label), xy=(x, 1.15), fontsize=12)

    # Set margins for the axes so that nodes aren't clipped
    ax.margins(0.15)
    ax.axis("off")

    return ax
