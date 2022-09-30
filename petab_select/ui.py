import csv
import os.path
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import petab

from .candidate_space import CandidateSpace
from .constants import (
    ESTIMATE,
    INITIAL_MODEL_METHODS,
    TYPE_PATH,
    Criterion,
    Method,
)
from .model import Model, default_compare
from .problem import Problem


def candidates(
    problem: Problem,
    candidate_space: Optional[CandidateSpace] = None,
    previous_predecessor_model: Optional[Model] = None,
    limit: Union[float, int] = np.inf,
    limit_sent: Union[float, int] = np.inf,
    calibrated_models: Optional[Dict[str, Model]] = None,
    newly_calibrated_models: Optional[Dict[str, Model]] = None,
    excluded_models: Optional[List[Model]] = None,
    excluded_model_hashes: Optional[List[str]] = None,
    criterion: Optional[Criterion] = None,
) -> CandidateSpace:
    """Search the model space for candidate models.

    A predecessor model is chosen from `newly_calibrated_models` if available,
    otherwise from `calibrated_models`, and is used for applicable methods.

    Args:
        problem:
            A PEtab Select problem.
        candidate_space:
            The candidate space. Defaults to a new candidate space based on the method
            defined in the problem.
        previous_predecessor_model:
            The previous predecessor model for a compatible method. This is
            used as the predecessor model for the current iteration, if a
            better model doesn't exist in the candidate space models.
        limit:
            The maximum number of models to add to the candidate space.
        limit_sent:
            The maximum number of models sent to the candidate space (which are possibly
            rejected and excluded).
        calibrated_models:
            All calibrated models in the model selection.
        newly_calibrated_models:
            All calibrated models in the most recent iteration of model
            selection.
        excluded_models:
            Models that will be excluded from model subspaces during the search for
            candidates.
        excluded_model_hashes:
            Hashes of models that will be excluded from model subspaces during the
            search for candidates.
        criterion:
            The criterion by which models will be compared. Defaults to the criterion
            defined in the PEtab Select problem.

    Returns:
        The candidate space, which contains the candidate models.
    """
    # FIXME might be difficult for a CLI tool to specify a specific predecessor
    #       model if their candidate space has models. Need a way to empty
    #       the candidate space of models... might be difficult with pickled
    #       candidate space objects/arguments?
    if excluded_models is None:
        excluded_models = []
    if excluded_model_hashes is None:
        excluded_model_hashes = []
    if calibrated_models is None:
        calibrated_models = {}
    if newly_calibrated_models is None:
        newly_calibrated_models = {}
    calibrated_models.update(newly_calibrated_models)
    if criterion is None:
        criterion = problem.criterion
    if candidate_space is None:
        candidate_space = problem.new_candidate_space(limit=limit)
    candidate_space.exclude_hashes(calibrated_models)

    # Set the predecessor model to the previous predecessor model.
    # Set the new predecessor_model from the initial model or
    # by calling ui.best to find the best model to jump to if
    # this is not the first step of the search.
    predecessor_model = previous_predecessor_model
    if newly_calibrated_models:
        predecessor_model = problem.get_best(
            newly_calibrated_models.values(),
            criterion=criterion,
        )
        # If the new predecessor model isn't better than the previous one,
        # keep the previous one.
        # If FAMoS jumped this will not be useful, since the jumped-to model
        # can be expected to be worse than the jumped-from model, in general.
        if not default_compare(
            model0=previous_predecessor_model,
            model1=predecessor_model,
            criterion=criterion,
        ):
            predecessor_model = previous_predecessor_model

        candidate_space.update_after_calibration(
            calibrated_models=calibrated_models,
            newly_calibrated_models=newly_calibrated_models,
            criterion=criterion,
        )
        # If candidate space not Famos then ignored.
        # Else, in case we jumped to most distant in this iteration, go into
        # calibration with only the model we've jumped to.
        if (
            candidate_space.governing_method == Method.FAMOS
            and candidate_space.jumped_to_most_distant
        ):
            return candidate_space

    if (
        predecessor_model is None
        and candidate_space.method in INITIAL_MODEL_METHODS
        and calibrated_models
    ):
        predecessor_model = problem.get_best(
            models=calibrated_models.values(),
            criterion=criterion,
        )
    if predecessor_model is not None:
        candidate_space.reset(predecessor_model)

    # TODO support excluding model IDs? should be faster but may have issues, e.g.:
    #      - duplicate model IDs among multiple model subspaces
    #      - perhaps less portable if model IDs are generated differently on different
    #        computers
    problem.model_space.exclude_models(models=excluded_models)
    problem.model_space.exclude_model_hashes(
        model_hashes=excluded_model_hashes
    )
    problem.model_space.search(candidate_space, limit=limit_sent)

    write_summary_tsv(
        problem=problem,
        candidate_space=candidate_space,
        previous_predecessor_model=previous_predecessor_model,
        predecessor_model=predecessor_model,
    )

    return candidate_space


def model_to_petab(
    model: Model,
    output_path: Optional[TYPE_PATH] = None,
) -> Dict[str, Union[petab.Problem, TYPE_PATH]]:
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
    models: List[Model],
    output_path_prefix: Optional[List[TYPE_PATH]] = None,
) -> List[Dict[str, Union[petab.Problem, TYPE_PATH]]]:
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


def best(
    problem: Problem,
    models: List[Model],
    criterion: Optional[Union[str, None]] = None,
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
    candidate_space: Optional[CandidateSpace] = None,
    previous_predecessor_model: Optional[Union[str, Model]] = None,
    predecessor_model: Optional[Model] = None,
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
            candidate_parameter_ids.symmetric_difference(
                predecessor_parameter_ids
            )
        )

    # FIXME remove once MostDistantCandidateSpace exists...
    method = candidate_space.method
    if (
        candidate_space.governing_method == Method.FAMOS
        and candidate_space.predecessor_model.predecessor_model_hash is None
    ):
        with open(candidate_space.summary_tsv, 'r') as f:
            if sum(1 for _ in f) > 1:
                method = Method.MOST_DISTANT

    candidate_space.write_summary_tsv(
        [
            method,
            len(candidate_space.models),
            diff_parameter_ids,
            predecessor_criterion,
            predecessor_parameter_ids,
            diff_candidates_parameter_ids,
        ]
    )
