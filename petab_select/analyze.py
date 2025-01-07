"""Methods to analyze results of model selection."""

import warnings
from collections.abc import Callable

import networkx as nx
import numpy as np

from .constants import Criterion
from .model import Model, ModelHash, default_compare
from .models import Models

__all__ = [
    # "get_predecessor_models",
    "group_by_predecessor_model",
    "group_by_iteration",
    "get_best_by_iteration",
    "compute_weights",
    "get_graph",
    "get_parameter_changes",
]


# def get_predecessor_models(models: Models) -> Models:
#    """Get all models that were predecessors to other models.
#
#    Args:
#        models:
#            The models
#
#    Returns:
#        The predecessor models.
#    """
#    predecessor_models = Models([
#        models.get(
#            model.predecessor_model_hash,
#            # Handle virtual initial model.
#            model.predecessor_model_hash,
#        ) for model in models
#    ])
#    return predecessor_models


def group_by_predecessor_model(models: Models) -> dict[ModelHash, Models]:
    """Group models by their predecessor model.

    Args:
        models:
            The models.

    Returns:
        Key is predecessor model hash, value is models.
    """
    result = {}
    for model in models:
        if model.predecessor_model_hash not in result:
            result[model.predecessor_model_hash] = Models()
        result[model.predecessor_model_hash].append(model)
    return result


def group_by_iteration(
    models: Models, sort: bool = True
) -> dict[int | None, Models]:
    """Group models by their iteration.

    Args:
        models:
            The models.
        sort:
            Whether to sort the iterations.

    Returns:
        Key is iteration, value is models.
    """
    result = {}
    for model in models:
        if model.iteration not in result:
            result[model.iteration] = Models()
        result[model.iteration].append(model)
    if sort:
        result = {iteration: result[iteration] for iteration in sorted(result)}
    return result


def get_best(
    models: Models,
    criterion: Criterion,
    compare: Callable[[Model, Model], bool] | None = None,
    compute_criterion: bool = False,
) -> Model:
    """Get the best model.

    Args:
        models:
            The models.
        criterion.
            The criterion.
        compare:
            The method used to compare two models.
            Defaults to :func:``petab_select.model.default_compare``.
        compute_criterion:
            Whether to try computing criterion values, if sufficient
            information is available (e.g., likelihood and number of
            parameters, to compute AIC).

    Returns:
        The best model.
    """
    if compare is None:
        compare = default_compare

    best_model = None
    for model in models:
        if compute_criterion and not model.has_criterion(criterion):
            model.get_criterion(criterion)
        if not model.has_criterion(criterion):
            warnings.warn(
                f"The model `{model.hash}` has no value set for criterion "
                f"`{criterion}`. Consider using `compute_criterion=True` "
                "if there is sufficient information already stored in the "
                "model (e.g. the likelihood).",
                RuntimeWarning,
                stacklevel=2,
            )
            continue
        if best_model is None:
            best_model = model
            continue
        if compare(best_model, model, criterion=criterion):
            best_model = model
    if best_model is None:
        raise KeyError(
            "None of the supplied models have a value set for the criterion "
            f"`{criterion}`."
        )
    return best_model


def get_best_by_iteration(
    models: Models,
    *args,
    **kwargs,
) -> dict[int, Models]:
    """Get the best model of each iteration.

    See :func:``get_best`` for additional required arguments.

    Args:
        models:
            The models.
        *args, **kwargs:
            Forwarded to :func:``get_best``.

    Returns:
        The strictly improving models. Keys are iteration, values are models.
    """
    iterations_models = group_by_iteration(models=models)
    best_by_iteration = {
        iteration: get_best(
            *args,
            models=iteration_models,
            **kwargs,
        )
        for iteration, iteration_models in iterations_models.items()
    }
    return best_by_iteration


def compute_weights(
    models: Models,
    criterion: Criterion,
    as_dict: bool = False,
) -> list[float] | dict[ModelHash, float]:
    """Compute criterion weights.

    N.B.: regardless of the criterion, the formula used is the Akaike weights
    formula, but with ``criterion`` values instead of the AIC.

    Args:
        models:
            The models.
        criterion:
            The criterion.
        as_dict:
            Whether to return a dictionary, with model hashes for keys.

    Returns:
        The criterion weights.
    """
    relative_criterion_values = np.array(
        models.get_criterion(criterion=criterion, relative=True)
    )
    weights = np.exp(-0.5 * relative_criterion_values)
    weights /= weights.sum()
    weights = weights.tolist()
    if as_dict:
        weights = dict(zip(models.hashes, weights, strict=False))
    return weights


def get_graph(
    models: Models,
    labels: dict[ModelHash, str] = None,
) -> nx.DiGraph:
    """Get a graph representation of the models in terms of their ancestry.

    Edges connect models with their predecessor models.

    Args:
        models:
            The models.
        labels:
            Alternative labels for the models. Keys are model hashes, values
            are the labels.

    Returns:
        The graph.
    """
    if labels is None:
        labels = {}

    G = nx.DiGraph()
    edges = []
    for model in models:
        tail = labels.get(
            model.predecessor_model_hash, model.predecessor_model_hash
        )
        head = labels.get(model.hash, model.hash)
        edges.append((tail, head))
    G.add_edges_from(edges)
    return G


def get_parameter_changes(
    models: Models,
    as_dict: bool = False,
) -> (
    dict[ModelHash, list[tuple[set[str], set[str]]]]
    | list[tuple[set[str], set[str]]]
):
    """Get the differences in parameters betweem models and their predecessors.

    Args:
        models:
            The models.
        as_dict:
            Whether to return a dictionary, with model hashes for keys.

    Returns:
        The parameter changes. Each model has a 2-tuple of sets of parameters.
        The first and second sets are the added and removed parameters,
        respectively. If the predecessor model is undefined (e.g. the
        ``VIRTUAL_INITIAL_MODEL``), then both sets will be empty.
    """
    estimated_parameters = {
        model.hash: set(model.estimated_parameters) for model in models
    }
    parameter_changes = [
        (set(), set())
        if model.predecessor_model_hash not in estimated_parameters
        else (
            estimated_parameters[model.hash].difference(
                estimated_parameters[model.predecessor_model_hash]
            ),
            estimated_parameters[model.predecessor_model_hash].difference(
                estimated_parameters[model.hash]
            ),
        )
        for model in models
    ]

    if as_dict:
        return dict(zip(models.hashes, parameter_changes, strict=True))
    return parameter_changes
