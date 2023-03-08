"""Visualization routines for model selection with pyPESTO."""
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

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


def selected_models(
    models: List[Model],
    criterion: Criterion,
    relative: str = True,
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

    zero = 0
    if relative:
        zero = models[-1].get_criterion(criterion)

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
        - zero
        for model in models
    }

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
        criterion + ('(relative)' if relative else '(absolute)'), fontsize=fz
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


def directed_graph(
    models: List[Model],
    criterion: Criterion = None,
    optimal_distance: float = 1,
    options: Dict = None,
    labels: Dict[str, str] = None,
    ax: plt.Axes = None,
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

    if labels is None:
        labels = {
            model_hash: model.model_id
            + (
                f"\n{model.get_criterion(criterion):.2f}"
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


def bar_graph(
    models: List[Model],
    criterion: Criterion = None,
    labels: Dict[str, str] = None,
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
    ax.bar(criterion_values.keys(), criterion_values.values())
    ax.set_xlabel("Model labels")
    ax.set_ylabel(criterion.value)

    return ax
