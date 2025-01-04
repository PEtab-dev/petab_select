"""The PEtab Select command-line interface."""

from pathlib import Path
from typing import Any

import click
import dill
import numpy as np
import pandas as pd
import yaml
from more_itertools import one

from . import ui
from .candidate_space import CandidateSpace
from .constants import (
    CANDIDATE_SPACE,
    MODELS,
    PETAB_YAML,
    PROBLEM,
    TERMINATE,
    UNCALIBRATED_MODELS,
)
from .model import ModelHash
from .models import Models, models_to_yaml_list
from .problem import Problem


def read_state(filename: str) -> dict[str, Any]:
    with open(filename, "rb") as f:
        state = dill.load(f)

    state[PROBLEM] = dill.loads(state[PROBLEM])
    state[CANDIDATE_SPACE] = dill.loads(state[CANDIDATE_SPACE])

    return state


def write_state(
    state: dict[str, Any],
    filename: str,
) -> dict[str, Any]:
    with open(filename, "wb") as f:
        dill.dump(state, f)


def get_state(
    problem: Problem,
    candidate_space: CandidateSpace,
) -> dict[str, Any]:
    state = {
        PROBLEM: dill.dumps(problem),
        CANDIDATE_SPACE: dill.dumps(candidate_space),
    }
    return state


@click.group()
def cli():
    pass


@cli.command("start_iteration")
@click.option(
    "--problem",
    "-p",
    "problem_yaml",
    help="The PEtab Select YAML problem file.",
)
@click.option(
    "--state",
    "-s",
    "state_dill",
    type=str,
    help="The file that stores the state.",
)
@click.option(
    "--output-uncalibrated-models",
    "-u",
    "uncalibrated_models_yaml",
    type=str,
    help="The file where uncalibrated models from this iteration will be stored.",
)
@click.option(
    "--method",
    "-m",
    "method",
    type=str,
    default=None,
    help="The method used to identify the candidate models. Defaults to the method in the problem YAML.",
)
@click.option(
    "--limit",
    "-l",
    "limit",
    type=float,
    default=np.inf,
    help="(Optional) Limit the number of models in the output.",
)
@click.option(
    "--limit-sent",
    "-L",
    "limit_sent",
    type=float,
    default=np.inf,
    help=(
        "(Optional) Limit the number of models sent to the candidate space "
        "(which are possibly rejected and excluded from the output)."
    ),
)
@click.option(
    "--relative-paths/--absolute-paths",
    "relative_paths",
    type=bool,
    default=False,
    help="Whether to output paths relative to the output file.",
)
@click.option(
    "--excluded-models",
    "-e",
    "excluded_model_files",
    type=str,
    multiple=True,
    default=None,
    help="Exclude models in this file.",
)
@click.option(
    "--excluded-model-hashes",
    "-E",
    "excluded_model_hash_files",
    type=str,
    multiple=True,
    default=None,
    help="Exclude model hashes in this file (one model hash per line).",
)
def start_iteration(
    problem_yaml: str,
    state_dill: str,
    uncalibrated_models_yaml: str,
    method: str = None,
    limit: float = np.inf,
    limit_sent: float = np.inf,
    relative_paths: bool = False,
    excluded_model_files: list[str] = None,
    excluded_model_hash_files: list[str] = None,
) -> None:
    """Search for candidate models in the model space.

    Documentation for arguments can be viewed with
    `petab_select start_iteration --help`.
    """
    problem = Problem.from_yaml(problem_yaml)
    if method is None:
        method = problem.method

    # `petab_select.ui.start_iteration` uses `petab_select.Problem.method` to
    # generate the candidate space.
    problem.method = method
    candidate_space = problem.new_candidate_space(limit=limit)

    # Setup state
    if not Path(state_dill).exists():
        Path(state_dill).parent.mkdir(parents=True, exist_ok=True)
    else:
        state = read_state(state_dill)
        if state["problem"].method != problem.method:
            raise NotImplementedError(
                "Changing method in the middle of a run is currently not "
                "supported. Delete the state to start with a new method."
            )
        problem = state["problem"]
        candidate_space = state["candidate_space"]

    excluded_models = Models()
    # TODO seems like default is `()`, not `None`...
    if excluded_model_files is not None:
        for models_yaml in excluded_model_files:
            excluded_models.extend(Models.from_yaml(models_yaml))

    # TODO test
    excluded_model_hashes = []
    if excluded_model_hash_files is not None:
        for excluded_model_hash_file in excluded_model_hash_files:
            with open(excluded_model_hash_file) as f:
                excluded_model_hashes += f.read().split("\n")

    excluded_hashes = [
        excluded_model.hash for excluded_model in excluded_models
    ]
    excluded_hashes += [
        ModelHash.from_hash(hash_str) for hash_str in excluded_model_hashes
    ]

    result = ui.start_iteration(
        problem=problem,
        candidate_space=candidate_space,
        limit=limit,
        limit_sent=limit_sent,
        excluded_hashes=excluded_hashes,
    )

    # Save state
    write_state(
        state=get_state(
            problem=problem,
            candidate_space=candidate_space,
        ),
        filename=state_dill,
    )

    # Save candidate models
    result[UNCALIBRATED_MODELS].to_yaml(
        filename=uncalibrated_models_yaml,
        relative_paths=relative_paths,
    )


@cli.command("end_iteration")
@click.option(
    "--state",
    "-s",
    "state_dill",
    type=str,
    help="The file that stores the state.",
)
@click.option(
    "--output-models",
    "-m",
    "models_yaml",
    type=str,
    help="The file where this iteration's calibrated models will be stored.",
)
@click.option(
    "--output-metadata",
    "-d",
    "metadata_yaml",
    type=str,
    help="The file where this iteration's metadata will be stored.",
)
@click.option(
    "--calibrated-models",
    "-c",
    "calibrated_models_yamls",
    type=str,
    multiple=True,
    help=(
        "The calibration results for the uncalibrated models of this iteration."
    ),
)
@click.option(
    "--relative-paths/--absolute-paths",
    "relative_paths",
    type=bool,
    default=False,
    help="Whether to output paths relative to the output file.",
)
def end_iteration(
    state_dill: str,
    models_yaml: str,
    metadata_yaml: str,
    calibrated_models_yamls: list[str] = None,
    relative_paths: bool = False,
) -> None:
    """Finalize a model selection iteration.

    Documentation for arguments can be viewed with
    `petab_select end_iteration --help`.
    """
    # Setup state
    state = read_state(state_dill)
    problem = state["problem"]
    candidate_space = state["candidate_space"]

    calibrated_models = Models()
    if calibrated_models_yamls:
        for calibrated_models_yaml in calibrated_models_yamls:
            calibrated_models.extend(Models.from_yaml(calibrated_models_yaml))

    # Finalize iteration results
    iteration_results = ui.end_iteration(
        problem=problem,
        candidate_space=candidate_space,
        calibrated_models=calibrated_models,
    )

    # Save iteration results
    ## State
    write_state(
        state=get_state(
            problem=problem,
            candidate_space=iteration_results[CANDIDATE_SPACE],
        ),
        filename=state_dill,
    )
    ## Models
    models_to_yaml_list(
        models=iteration_results[MODELS],
        output_yaml=models_yaml,
        relative_paths=relative_paths,
    )
    ## Metadata
    metadata = {
        TERMINATE: iteration_results[TERMINATE],
    }
    with open(metadata_yaml, "w") as f:
        yaml.dump(metadata, f)


@cli.command("model_to_petab")
@click.option(
    "--model",
    "-m",
    "models_yamls",
    multiple=True,
    help="The PEtab Select model YAML file.",
)
@click.option(
    "--output",
    "-o",
    "output_path",
    type=str,
    help="The directory where the PEtab files will be output.",
)
@click.option(
    "--model_id",
    "-i",
    "model_id",
    type=str,
    default=None,
    help=(
        "(Optional) The ID of the model to use, in case the YAML file "
        "contains multiple models."
    ),
)
def model_to_petab(
    models_yamls: list[str],
    output_path: str,
    model_id: str = None,
) -> None:
    """Create a PEtab problem from a PEtab Select model YAML file.

    The filename for the PEtab problem YAML file is output to `stdout`.

    Documentation for arguments can be viewed with
    `petab_select model_to_petab --help`.
    """
    models = Models()
    for models_yaml in models_yamls:
        models.extend(Models.from_yaml(models_yaml))

    model0 = None
    try:
        model0 = one(models)
    except:
        for model in models:
            if model.model_id == model_id:
                if model0 is not None:
                    raise ValueError("There are multiple models with this ID.")
                model0 = model
                # TODO could `break` here and remove the above `ValueError`
                #      and the `model0` logic
    if model0 is None:
        raise ValueError("Could not find a model with the specified model ID.")

    if model_id is not None and model0.model_id != model_id:
        raise ValueError(
            "The ID of the model from the YAML file does not match the "
            "specified ID."
        )

    result = ui.model_to_petab(model0, output_path)
    print(result[PETAB_YAML])


@cli.command("models_to_petab")
@click.option(
    "--models",
    "-m",
    "models_yamls",
    type=str,
    multiple=True,
    help="The PEtab Select model YAML file, containing a list of models.",
)
@click.option(
    "--output",
    "-o",
    "output_path_prefix",
    type=str,
    help="The directory where the PEtab files will be output. The PEtab files will be stored in a model-specific subdirectory.",
)
def models_to_petab(
    models_yamls: list[str],
    output_path_prefix: str,
) -> None:
    """Create a PEtab problem for each model in a PEtab Select model YAML file.

    NB: Models in the YAML file must have a model ID.

    The output to `stdout` is a two-column tab-separated list, where the first
    column is the model ID, and the second column is the location of the PEtab
    problem YAML file for that model.

    Documentation for arguments can be viewed with
    `petab_select models_to_petab --help`.
    """
    models = Models()
    for models_yaml in models_yamls:
        models.extend(Models.from_yaml(models_yaml))

    model_ids = pd.Series([model.model_id for model in models])
    duplicates = "\n".join(set(model_ids[model_ids.duplicated()]))
    if duplicates:
        raise ValueError(
            "It appears that the provided PEtab Select model YAML file "
            "contains multiple models with the same ID. The following "
            f"duplicates were detected: {duplicates}"
        )

    results = ui.models_to_petab(
        models,
        output_path_prefix=output_path_prefix,
    )
    result_string = "\n".join(
        [
            "\t".join([model.model_id, result[PETAB_YAML]])
            for model, result in zip(models, results, strict=False)
        ]
    )
    print(result_string)


@cli.command("get_best")
@click.option(
    "--problem",
    "-p",
    "problem_yaml",
    type=str,
    help="The PEtab Select YAML problem file.",
)
@click.option(
    "--models",
    "-m",
    "models_yamls",
    type=str,
    multiple=True,
    help="A list of calibrated models.",
)
@click.option(
    "--output",
    "-o",
    "output",
    type=str,
    help="The file where the best model will be stored.",
)
@click.option(
    "--state",
    "-s",
    "state_filename",
    type=str,
    default=None,
    help="The file that stores the state.",
)
@click.option(
    "--criterion",
    "-c",
    "criterion",
    type=str,
    default=None,
    help="The criterion by which models will be compared.",
)
@click.option(
    "--relative-paths/--absolute-paths",
    "relative_paths",
    type=bool,
    default=False,
    help="Whether to output paths relative to the output file.",
)
def get_best(
    problem_yaml: str,
    models_yamls: list[str],
    output: str,
    state_filename: str = None,
    criterion: str = None,
    relative_paths: bool = False,
) -> None:
    """Get the best model from a list of models.

    Documentation for arguments can be viewed with
    `petab_select get_best --help`.
    """
    paths_relative_to = None
    if relative_paths:
        paths_relative_to = Path(output).parent

    problem = Problem.from_yaml(problem_yaml)

    models = Models()
    for models_yaml in models_yamls:
        models.extend(Models.from_yaml(models_yaml))

    best_model = ui.get_best(
        problem=problem,
        models=models,
        criterion=criterion,
    )
    best_model.to_yaml(output)


cli.add_command(start_iteration)
cli.add_command(end_iteration)
cli.add_command(model_to_petab)
cli.add_command(models_to_petab)
cli.add_command(get_best)
