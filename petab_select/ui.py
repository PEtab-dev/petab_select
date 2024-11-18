import copy
from pathlib import Path
from typing import Any

import numpy as np
import petab.v1 as petab

from .candidate_space import CandidateSpace, FamosCandidateSpace
from .constants import (
    CANDIDATE_SPACE,
    MODELS,
    PREDECESSOR_MODEL,
    TERMINATE,
    TYPE_PATH,
    UNCALIBRATED_MODELS,
    VIRTUAL_INITIAL_MODEL,
    Criterion,
    Method,
)
from .model import Model, ModelHash, default_compare
from .problem import Problem

__all__ = [
    "start_iteration",
    "end_iteration",
    "model_to_petab",
    "models_to_petab",
    "get_best",
    "write_summary_tsv",
]


def get_iteration(candidate_space: CandidateSpace) -> dict[str, Any]:
    return {
        CANDIDATE_SPACE: candidate_space,
        UNCALIBRATED_MODELS: candidate_space.models,
        PREDECESSOR_MODEL: candidate_space.get_predecessor_model(),
    }


def start_iteration(
    problem: Problem,
    candidate_space: CandidateSpace | None = None,
    limit: float | int = np.inf,
    limit_sent: float | int = np.inf,
    excluded_hashes: list[ModelHash] | None = None,
    criterion: Criterion | None = None,
    user_calibrated_models: list[Model] | dict[ModelHash, Model] | None = None,
) -> CandidateSpace:
    """Search the model space for candidate models.

    The predecessor model can be specified in the `candidate_space`
    (:func:`CandidateSpace.set_predecessor_model). If `candidate_space` is not
    provided, then the predecessor model can be specified in `problem`
    (:attr:`Problem.candidate_space_arguments`).

    Args:
        problem:
            A PEtab Select problem.
        candidate_space:
            The candidate space. Defaults to a new candidate space based on the method
            defined in the problem.
        limit:
            The maximum number of models to add to the candidate space.
        limit_sent:
            The maximum number of models sent to the candidate space (which are possibly
            rejected and excluded).
        excluded_hashes:
            Hashes of models that will be excluded from the candidate space.
        criterion:
            The criterion by which models will be compared. Defaults to the criterion
            defined in the PEtab Select problem.
        user_calibrated_models:
            Models that were already calibrated by the user. When supplied as a
            `dict`, the keys are model hashes. If a model in the
            candidates has the same hash as a model in
            `user_calibrated_models`, then the candidate will be replaced with
            the calibrated version. Calibration tools will only receive uncalibrated
            models from this method.

    Returns:
        A dictionary, with the following items:
            :const:`petab_select.constants.CANDIDATE_SPACE`:
                The candidate space.
            :const:`petab_select.constants.MODELS`:
                The uncalibrated models of the current iteration.
    """
    """
    FIXME(dilpath)
    - currently takes predecessor model from
      candidate_space.previous_predecessor_model
    - deprecate limit_sent? possibly unused by anyone
    - add `Iteration` class to manage an iteration, append to
      `CandidateSpace.iterations`?
    """
    do_search = True
    # FIXME might be difficult for a CLI tool to specify a specific predecessor
    #       model if their candidate space has models. Need a way to empty
    #       the candidate space of models... might be difficult with pickled
    #       candidate space objects/arguments?
    if excluded_hashes is None:
        excluded_hashes = []
    if candidate_space is None:
        candidate_space = problem.new_candidate_space(limit=limit)

    if criterion is None:
        criterion = problem.criterion
    if criterion is None:
        raise ValueError("Please provide a criterion.")
    candidate_space.criterion = criterion

    # Set the predecessor model to the previous predecessor model.
    predecessor_model = candidate_space.previous_predecessor_model

    # If the predecessor model has not yet been calibrated, then calibrate it.
    if (
        predecessor_model is not None
        and predecessor_model != VIRTUAL_INITIAL_MODEL
    ):
        if (
            predecessor_model.get_criterion(
                criterion,
                raise_on_failure=False,
            )
            is None
        ):
            candidate_space.models = [copy.deepcopy(predecessor_model)]
            # Dummy zero likelihood, which the predecessor model will
            # improve on after it's actually calibrated.
            predecessor_model.set_criterion(Criterion.LH, 0.0)
            candidate_space.set_iteration_user_calibrated_models(
                user_calibrated_models=user_calibrated_models
            )
            return get_iteration(candidate_space=candidate_space)

        # Exclude the calibrated predecessor model.
        if not candidate_space.excluded(predecessor_model):
            candidate_space.set_excluded_hashes(
                predecessor_model,
                extend=True,
            )

    # Set the new predecessor_model from the initial model or
    # by calling ui.best to find the best model to jump to if
    # this is not the first step of the search.
    if candidate_space.latest_iteration_calibrated_models:
        predecessor_model = problem.get_best(
            candidate_space.latest_iteration_calibrated_models.values(),
            criterion=criterion,
        )
        # If the new predecessor model isn't better than the previous one,
        # keep the previous one.
        # If FAMoS jumped this will not be useful, since the jumped-to model
        # can be expected to be worse than the jumped-from model, in general.
        if not default_compare(
            model0=candidate_space.previous_predecessor_model,
            model1=predecessor_model,
            criterion=criterion,
        ):
            predecessor_model = candidate_space.previous_predecessor_model

        # If candidate space not Famos then ignored.
        # Else, in case we jumped to most distant in this iteration, go into
        # calibration with only the model we've jumped to.
        # TODO handle as proper `MostDistantCandidateSpace`
        if (
            isinstance(candidate_space, FamosCandidateSpace)
            and candidate_space.jumped_to_most_distant
        ):
            return get_iteration(candidate_space=candidate_space)

    if predecessor_model is not None:
        candidate_space.reset(predecessor_model)

    # FIXME store exclusions in candidate space only
    problem.model_space.exclude_model_hashes(model_hashes=excluded_hashes)
    while do_search:
        problem.model_space.search(candidate_space, limit=limit_sent)

        write_summary_tsv(
            problem=problem,
            candidate_space=candidate_space,
            previous_predecessor_model=candidate_space.previous_predecessor_model,
            predecessor_model=predecessor_model,
        )

        if candidate_space.models:
            break

        # No models were found. Repeat the search with the same candidate space,
        # if the candidate space is able to switch methods.
        # N.B.: candidate spaces that switch methods must raise `StopIteration`
        # when they stop switching.
        if isinstance(candidate_space, FamosCandidateSpace):
            try:
                candidate_space.update_after_calibration(
                    iteration_calibrated_models={},
                )
                continue
            except StopIteration:
                break

        # No models were found, and the method doesn't switch, so no further
        # models can be found.
        break

    candidate_space.previous_predecessor_model = predecessor_model

    candidate_space.set_iteration_user_calibrated_models(
        user_calibrated_models=user_calibrated_models
    )
    return get_iteration(candidate_space=candidate_space)


def end_iteration(
    candidate_space: CandidateSpace,
    calibrated_models: list[Model] | dict[str, Model],
) -> dict[str, dict[ModelHash, Model] | bool | CandidateSpace]:
    """Finalize model selection iteration.

    All models from the current iteration are provided to the calibration tool.
    This includes user-calibrated models that the tool did not see until now.

    A termination signal is also provided, if the model selection algorithm
    ends.

    Args:
        candidate_space:
            The candidate space.
        calibrated_models:
            The calibration results for the uncalibrated models of this
            iteration.

    Returns:
        A dictionary, with the following items:
            :const:`petab_select.constants.MODELS`:
                All calibrated models for the current iteration as a
                dictionary, where keys are model hashes, and values are models.
            :const:`petab_select.constants.TERMINATE`:
                Whether PEtab Select has decided to end the model selection,
                as a boolean.
    """
    if isinstance(calibrated_models, list):
        calibrated_models = {
            model.get_hash(): model for model in calibrated_models
        }

    iteration_results = {
        MODELS: candidate_space.get_iteration_calibrated_models(
            calibrated_models=calibrated_models,
            reset=True,
        )
    }

    terminate = not iteration_results[MODELS]
    try:
        candidate_space.update_after_calibration(
            iteration_calibrated_models=iteration_results[MODELS],
        )
    except StopIteration:
        # e.g. FAMoS switch_method encountered "None", indicating end of model
        # selection
        terminate = True
    iteration_results[TERMINATE] = terminate

    iteration_results[CANDIDATE_SPACE] = candidate_space

    return iteration_results


def model_to_petab(
    model: Model,
    output_path: TYPE_PATH | None = None,
) -> dict[str, petab.Problem | TYPE_PATH]:
    """Generate the PEtab problem for a model.

    Args:
        model:
            The model.
        output_path:
            If specified, the PEtab problem will be output to files in this directory.

    Returns:
        The PEtab problem, and the path to the PEtab problem YAML file, if an output
        path is provided.
    """
    return model.to_petab(output_path=output_path)


def models_to_petab(
    models: list[Model],
    output_path_prefix: list[TYPE_PATH] | None = None,
) -> list[dict[str, petab.Problem | TYPE_PATH]]:
    """Generate the PEtab problems for a list of models.

    Args:
        models:
            The list of model.
        output_path_prefix:
            If specified, the PEtab problem will be output to files in subdirectories
            of this path, where each subdirectory corresponds to a model.

    Returns:
        The PEtab problems, and the paths to the PEtab problem YAML files, if an output
        path prefix is provided.
    """
    output_path_prefix = Path(output_path_prefix)
    result = []
    for model in models:
        output_path = output_path_prefix / model.model_id
        result.append(model_to_petab(model=model, output_path=output_path))
    return result


def get_best(
    problem: Problem,
    models: list[Model],
    criterion: str | None | None = None,
) -> Model:
    """Get the best model from a list of models.

    Args:
        problem:
            The PEtab Select problem.
        models:
            The list of models.
        criterion:
            The criterion by which models will be compared. Defaults to
            `problem.criterion`.

    Returns:
        The best model.
    """
    # TODO return list, when multiple models are equally "best"
    return problem.get_best(models=models, criterion=criterion)


def write_summary_tsv(
    problem: Problem,
    candidate_space: CandidateSpace | None = None,
    previous_predecessor_model: str | Model | None = None,
    predecessor_model: Model | None = None,
) -> None:
    if candidate_space.summary_tsv is None:
        return

    previous_predecessor_parameter_ids = set()
    if isinstance(previous_predecessor_model, Model):
        previous_predecessor_parameter_ids = set(
            previous_predecessor_model.get_estimated_parameter_ids_all()
        )

    if predecessor_model is None:
        predecessor_model = candidate_space.predecessor_model
    predecessor_parameter_ids = set()
    predecessor_criterion = None
    if isinstance(predecessor_model, Model):
        predecessor_parameter_ids = set(
            predecessor_model.get_estimated_parameter_ids_all()
        )
        predecessor_criterion = predecessor_model.get_criterion(
            problem.criterion
        )

    diff_parameter_ids = (
        previous_predecessor_parameter_ids.symmetric_difference(
            predecessor_parameter_ids
        )
    )

    diff_candidates_parameter_ids = []
    for candidate_model in candidate_space.models:
        candidate_parameter_ids = set(
            candidate_model.get_estimated_parameter_ids_all()
        )
        diff_candidates_parameter_ids.append(
            list(
                candidate_parameter_ids.symmetric_difference(
                    predecessor_parameter_ids
                )
            )
        )

    # FIXME remove once MostDistantCandidateSpace exists...
    method = candidate_space.method
    if (
        isinstance(candidate_space, FamosCandidateSpace)
        and isinstance(candidate_space.predecessor_model, Model)
        and candidate_space.predecessor_model.predecessor_model_hash is None
    ):
        with open(candidate_space.summary_tsv) as f:
            if sum(1 for _ in f) > 1:
                method = Method.MOST_DISTANT

    candidate_space.write_summary_tsv(
        [
            method,
            len(candidate_space.models),
            sorted(diff_parameter_ids),
            predecessor_criterion,
            sorted(predecessor_parameter_ids),
            sorted(
                diff_candidates_parameter_ids,
                key=lambda x: [x[i] for i in range(len(x))],
            ),
        ]
    )
