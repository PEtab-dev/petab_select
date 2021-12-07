"""The PEtab Select command-line interface."""
from pathlib import Path
from typing import List
import yaml

import click
import dill
import numpy as np
import pandas as pd

from . import ui
from .constants import (
    INITIAL_MODEL_METHODS,
    PETAB_YAML,
)
from .candidate_space import (
    BackwardCandidateSpace,
    BruteForceCandidateSpace,
    CandidateSpace,
    ForwardCandidateSpace,
    LateralCandidateSpace,
    method_to_candidate_space_class,
)
from .model import (
    Model,
    models_from_yaml_list,
)
from .problem import Problem


@click.group()
def cli():
    pass


@cli.command("candidates")
@click.option(
    '--yaml',
    '-y',
    'yaml_',
    help='The PEtab Select YAML problem file.',
)
@click.option(
    '--state',
    '-s',
    'state',
    type=str,
    help='The file that stores the state.',
)
@click.option(
    '--output',
    '-o',
    'output',
    type=str,
    help='The file where candidate models will be stored.',
)
@click.option(
    '--method',
    '-m',
    'method',
    type=str,
    default=None,
    help='The method used to identify the candidate models. Defaults to the method in the problem YAML.',  # noqa: E501
)
@click.option(
    '--initial',
    '-i',
    'initial',
    type=str,
    default=None,
    help='(Optional) The initial model used in the candidate model search.',
)
@click.option(
    '--best',
    '-b',
    'best',
    type=str,
    default=None,
    help=(
        '(Optional) Initialize with the best model from a collection of '
        'calibrated models.'
    ),
)
@click.option(
    '--limit',
    '-l',
    'limit',
    type=float,
    default=np.inf,
    help='(Optional) Limit the number of models in the output.',
)
@click.option(
    '--relative-paths/--absolute-paths',
    'relative_paths',
    type=bool,
    default=False,
    help='Whether to output paths relative to the output file.',
)
@click.option(
    '--exclude-models',
    '-e',
    'exclude_models',
    type=str,
    multiple=True,
    default=None,
    help='Exclude models in this file.',
)
def candidates(
    yaml_: str,
    state: str,
    output: str,
    method: str = None,
    initial: str = None,
    best: str = None,
    limit: float = np.inf,
    relative_paths: bool = False,
    exclude_models: List[str] = None,
) -> None:
    """Search for candidate models in the model space.

    Documentation for arguments can be viewed with
    `petab_select candidates --help`.
    """
    if initial is not None and best is not None:
        raise KeyError(
            'The `initial` (`-i`) and `best` (`-b`) arguments cannot be used '
            'together, as they both set the initial model.'
        )

    paths_relative_to = None
    if relative_paths:
        paths_relative_to = Path(output).parent

    problem = Problem.from_yaml(yaml_)
    if method is None:
        method = problem.method

    # `petab_select.ui.candidates` uses `petab_select.Problem.method` to generate
    # the candidate space.
    problem.method = method

    if Path(state).exists():
        # Load state
        with open(state, 'rb') as f:
            problem.set_state(dill.load(f))
    else:
        # Create the output path for the state
        Path(state).parent.mkdir(parents=True, exist_ok=True)

    excluded_models = []
    if exclude_models is not None:
        for model_yaml_list in exclude_models:
            excluded_models.extend(models_from_yaml_list(model_yaml_list))

    initial_model = None
    if best is not None:
        calibrated_models = models_from_yaml_list(best)
        initial_model = problem.get_best(calibrated_models)

    if initial is not None:
        initial_model = Model.from_yaml(initial)

    candidate_space = ui.candidates(
        problem=problem,
        initial_model=initial_model,
        limit=limit,
        excluded_models=excluded_models,
    )

    # Save state
    with open(state, 'wb') as f:
        dill.dump(problem.get_state(), f)

    model_dicts = [
        model.to_dict(paths_relative_to=paths_relative_to)
        for model in candidate_space.models
    ]
    model_dicts = None if not model_dicts else model_dicts
    # Save candidates
    with open(output, 'w') as f:
        yaml.dump(model_dicts, f)


@cli.command("model_to_petab")
@click.option(
    '--yaml',
    '-y',
    'yaml_',
    help='The PEtab Select model YAML file.',
)
@click.option(
    '--output',
    '-o',
    'output_path',
    type=str,
    help='The directory where the PEtab files will be output.',
)
@click.option(
    '--model',
    '-m',
    'model_id',
    type=str,
    default=None,
    help=(
        '(Optional) The ID of the model to use, in case the YAML file '
        'contains multiple models.'
    ),
)
def model_to_petab(
    yaml_: str,
    output_path: str,
    model_id: str = None,
) -> None:
    """Create a PEtab problem from a PEtab Select model YAML file.

    The filename for the PEtab problem YAML file is output to `stdout`.

    Documentation for arguments can be viewed with
    `petab_select model_to_petab --help`.
    """
    try:
        model = Model.from_yaml(yaml_)
    except ValueError as e1:
        # There may be multiple models in a single file.
        if model_id is None:
            raise ValueError(
                'There may be multiple models defined in the provided PEtab '
                'Select model YAML file. If so, please specify the ID of one '
                'model that you would like to convert to PEtab, or use '
                '`models2petab` to convert all of them.'
            )
        try:
            models = models_from_yaml_list(yaml_)
            try:
                model = one([
                    model
                    for model in models
                    if model.model_id == model_id
                ])
            except ValueError:
                raise ValueError(
                    'There must be exactly one model with the provided model '
                    f'ID "{model_id}", in the provided PEtab Select model '
                    'YAML file. This may not be the case.'
                )
        except Exception as e2:
            # Unknown error.
            raise Exception([e1, e2])

    if model_id is not None and model.model_id != model_id:
        raise ValueError(
            'The ID of the model from the YAML file does not match the '
            'specified ID.'
        )

    result = ui.model_to_petab(model, output_path)
    print(result[PETAB_YAML])


@cli.command("models_to_petab")
@click.option(
    '--yaml',
    '-y',
    'yaml_',
    help='The PEtab Select model YAML file, containing a list of models.',
)
@click.option(
    '--output',
    '-o',
    'output_path_prefix',
    type=str,
    help='The directory where the PEtab files will be output. The PEtab files will be stored in a model-specific subdirectory.',
)
def models_to_petab(
    yaml_: str,
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
    models = models_from_yaml_list(yaml_)
    model_ids = pd.Series([model.model_id for model in models])
    duplicates = '\n'.join(set(model_ids[model_ids.duplicated()]))
    if duplicates:
        raise ValueError(
            'It appears that the provided PEtab Select model YAML file '
            'contains multiple models with the same ID. The following '
            f'duplicates were detected: {duplicates}'
        )

    results = ui.models_to_petab(
        models,
        output_path_prefix=output_path_prefix,
    )
    result_string = '\n'.join([
        '\t'.join([model.model_id, result[PETAB_YAML]])
        for model, result in zip(models, results)
    ])
    print(result_string)


@cli.command("best")
@click.option(
    '--yaml',
    '-y',
    'yaml_',
    help='The PEtab Select YAML problem file.',
)
@click.option(
    '--models_yaml',
    '-m',
    'models_yaml',
    type=str,
    help='A list of calibrated models.',
)
@click.option(
    '--output',
    '-o',
    'output',
    type=str,
    help='The file where the best model will be stored.',
)
@click.option(
    '--state',
    '-s',
    'state',
    type=str,
    default=None,
    help='The file that stores the state.',
)
@click.option(
    '--criterion',
    '-c',
    'criterion',
    type=str,
    default=None,
    help='The criterion by which models will be compared.',
)
@click.option(
    '--relative-paths/--absolute-paths',
    'relative_paths',
    type=bool,
    default=False,
    help='Whether to output paths relative to the output file.',
)
def best(
    yaml_: str,
    models_yaml: str,
    output: str,
    state: str = None,
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

    problem = Problem.from_yaml(yaml_)
    if state is not None and Path(state).exists():
        # Load state
        with open(state, 'rb') as f:
            problem.set_state(dill.load(f))

    calibrated_models = models_from_yaml_list(models_yaml)
    best_model = ui.best(problem=problem, models=calibrated_models, criterion=criterion)
    best_model.to_yaml(output, paths_relative_to=paths_relative_to)


cli.add_command(candidates)
cli.add_command(model_to_petab)
cli.add_command(models_to_petab)
cli.add_command(best)
