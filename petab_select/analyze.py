"""Methods to analyze results of model selection."""
import numpy as np

from collections.abc import Callable

from .constants import Criterion
from .model import Model, ModelHash, default_compare
from .models import Models

__all__ = [
    # "get_predecessor_models",
    "group_by_predecessor_model",
    "group_by_iteration",
    "get_best_by_iteration",
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
        if best_model is None:
            if model.has_criterion(criterion):
                best_model = model
            # TODO warn if criterion is not available?
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
    as_dict: bool = True,
) -> list[float] | dict[ModelHash, float]:
    """Compute criterion weights.

    Args:
        models:
            The models.
        criterion:
            The criterion.
        as_dict:
            Whether to return a dictionary, with model hashes for keys.

    Returns:
        The criterion weights.
    -------
    dict:
        Dictionary with model hashes as keys and weights as values.
    """
    relative_criterion_values = models.get_criterion(criterion=criterion, relative=True, as_dict=True)
    sum_of_weights = np.exp(-0.5*np.array(relative_criterion_values.values())).sum()
    weights = {
        model.get_hash(): np.exp(-0.5*relative_criterion_values[model.get_hash()]) / sum_of_weights
        for model_hash, relative_criterion_value in relative_criterion_values.items()
    }
    return weights