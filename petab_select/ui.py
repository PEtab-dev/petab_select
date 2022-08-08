from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import petab

from .candidate_space import CandidateSpace
from .constants import TYPE_PATH, VIRTUAL_INITIAL_MODEL_METHODS
from .model import Model
from .problem import Problem


def candidates(
    problem: Problem,
    candidate_space: Optional[CandidateSpace] = None,
    predecessor_model: Optional[Model] = None,
    limit: Union[float, int] = np.inf,
    limit_sent: Union[float, int] = np.inf,
    excluded_models: Optional[List[Model]] = None,
    excluded_model_hashes: Optional[List[str]] = None,
) -> CandidateSpace:
    """Search the model space for candidate models.

    Args:
        problem:
            A PEtab Select problem.
        candidate_space:
            The candidate space. Defaults to a new candidate space based on the method
            defined in the problem.
        predecessor_model:
            The predecessor model for a compatible method. Defaults to the best model in
            `Problem.calibrated_models`, if available.
        limit:
            The maximum number of models to add to the candidate space.
        limit_sent:
            The maximum number of models sent to the candidate space (which are possibly
            rejected and excluded).
        excluded_models:
            Models that will be excluded from model subspaces during the search for
            candidates.
        excluded_model_hashes:
            Hashes of models that will be excluded from model subspaces during the
            search for candidates.

    Returns:
        The candidate space, which contains the candidate models.
    """
    if candidate_space is None:
        candidate_space = problem.new_candidate_space(limit=limit)
        if problem.calibrated_models:
            candidate_space.exclude(problem.calibrated_models)
    if excluded_models is None:
        excluded_models = []
    if excluded_model_hashes is None:
        excluded_model_hashes = []
    if candidate_space.famos_to_csv_path:
        previous_predecessor_model = (
            candidate_space.inner_candidate_space.predecessor_model
        )

    if (
        predecessor_model is None
        and candidate_space.method in VIRTUAL_INITIAL_MODEL_METHODS
        and problem.calibrated_models
    ):
        predecessor_model = problem.get_best()
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

    if candidate_space.famos_to_csv_path:
        write_famos_progress_to_csv(
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


def write_famos_progress_to_csv(
    problem: Problem,
    candidate_space: Optional[CandidateSpace] = None,
    previous_predecessor_model: Optional[Model] = None,
    predecessor_model: Optional[Model] = None,
) -> None:
    import os.path
    import csv

    previous_parameters = previous_predecessor_model.get_parameter_values(
        parameter_ids=previous_predecessor_model.petab_parameters
    )
    previous_parameter_indices = [
        index + 1
        for index in range(len(previous_parameters))
        if previous_parameters[index] == 'estimate'
    ]
    current_parameters = predecessor_model.get_parameter_values(
        parameter_ids=predecessor_model.petab_parameters
    )
    current_parameter_indices = [
        index + 1
        for index in range(len(current_parameters))
        if current_parameters[index] == 'estimate'
    ]
    changed_parameters = set(previous_parameter_indices).symmetric_difference(
        set(current_parameter_indices)
    )

    candidate_model_changed_pars = []
    for candidate_model in candidate_space.models:
        candidate_model_pars = candidate_model.get_parameter_values(
            parameter_ids=predecessor_model.petab_parameters
        )
        candidate_model_indices = [
            index + 1
            for index in range(len(candidate_model_pars))
            if candidate_model_pars[index] == 'estimate'
        ]
        candidate_model_changed_pars.append(
            set(candidate_model_indices).symmetric_difference(
                current_parameter_indices
            )
        )

    if not os.path.exists(candidate_space.famos_to_csv_path):
        with open(
            candidate_space.famos_to_csv_path, 'w', encoding='UTF8'
        ) as f:
            writer = csv.writer(f)

            writer.writerow(
                [
                    'current method',
                    '#candidate models',
                    'previous change of parameters',
                    'current model criterion',
                    'current model',
                    'candidate models changed pars',
                ]
            )

    with open(candidate_space.famos_to_csv_path, 'a', encoding='UTF8') as f:
        writer = csv.writer(f)

        writer.writerow(
            [
                len(candidate_space.models),
                candidate_space.inner_candidate_space.method,
                changed_parameters,
                predecessor_model.get_criterion(problem.criterion),
                current_parameter_indices,
                candidate_model_changed_pars,
            ]
        )
