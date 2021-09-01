"""The PEtab Select command-line interface."""
from pathlib import Path
from typing import List
import yaml

import click
import dill
import pandas as pd

from .constants import (
    INITIAL_MODEL_METHODS,
)
from .candidate_space import (
    BackwardCandidateSpace,
    BruteForceCandidateSpace,
    CandidateSpace,
    ForwardCandidateSpace,
    LateralCandidateSpace,
)
from .model import (
    Model,
    models_from_yaml_list,
)
from .problem import Problem


@click.group()
def cli():
    pass


def parse_candidate_space(
    method: str,
    model0_yaml_path: str,
    model0: Model,
) -> CandidateSpace:
    """Generate an appropriate candidate space instance.

    Args:
        method:
            The method used to identify candidate models.
        model0_yaml_path:
            The location of a PEtab Select model YAML file, that will be used
            to initialize a search for candidates if applicable.

    Returns:
        An instance of a CandidateSpace subclass.
    """
    if (
        method in INITIAL_MODEL_METHODS
        and
        (
            model0 is None
            and
            model0_yaml_path is None
        )
    ):
        raise ValueError(
            f'Please supply an initial model when using the method "{method}".'
        )

    # Already checked in `candidates`, should never raise.
    if model0 is not None and model0_yaml_path is not None:
        raise KeyError(
            'Both an initial model and a path to an initial model file were '
            'provided.'
        )
    if model0_yaml_path is not None:
        model0 = Model.from_yaml(model0_yaml_path)

    if method == 'forward':
        candidate_space = ForwardCandidateSpace(model0)
    elif method == 'backward':
        candidate_space = BackwardCandidateSpace(model0)
    elif method == 'lateral':
        candidate_space = LateralCandidateSpace(model0)
    elif method == 'brute_force':
        candidate_space = BruteForceCandidateSpace(model0)
    else:
        raise ValueError(
            f'Unknown search method: {method}'
        )

    return candidate_space


@cli.command()
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
    help='The method used to identify the candidate models.',
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
    type=int,
    default=None,
    help='(Optional) Limit the number of models in the output.',
)
def candidates(
    yaml_: str,
    state: str,
    output: str,
    method: str,
    initial: str = None,
    best: str = None,
    limit: int = None,
) -> None:
    """Search for candidate models in the model space.

    Documentation for arguments can be viewed with
    `petab_select candidates --help`.
    """
    if initial is not None and best is not None:
        raise KeyError(
            'The `initial` (`-i`) and `best` (`-b`) arguments cannot be used '
            'together.'
        )

    problem = Problem.from_yaml(yaml_)
    if Path(state).exists():
        # Load state
        with open(state, 'rb') as f:
            problem.set_state(dill.load(f))
    else:
        # Create the output path for the state
        Path(state).parent.mkdir(parents=True, exist_ok=True)

    model0 = None
    if best is not None:
        calibrated_models = models_from_yaml_list(best)       
        model0 = problem.get_best(calibrated_models)

    candidate_space = parse_candidate_space(
        method=method,
        model0_yaml_path=initial,
        model0=model0,
    )

    models = problem.model_space.neighbors(candidate_space, limit=limit)
    # Save state
    with open(state, 'wb') as f:
        dill.dump(problem.get_state(), f)

    model_dicts = [
        model.to_dict()
        for model in models
    ]
    model_dicts = None if not model_dicts else model_dicts
    # Save candidates
    with open(output, 'w') as f:
        yaml.dump(model_dicts, f)


@cli.command()
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
def model2petab(
    yaml_: str,
    output_path: str,
    model_id: str = None,
) -> None:
    """Create a PEtab problem from a PEtab Select model YAML file.

    The filename for the PEtab problem YAML file is output to `stdout`.

    Documentation for arguments can be viewed with
    `petab_select model2petab --help`.
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

    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)
    _, petab_yaml = model.to_petab(output_path)
    print(petab_yaml)


@cli.command()
@click.option(
    '--yaml',
    '-y',
    'yaml_',
    help='The PEtab Select model YAML file, containing a list of models.',
)
@click.option(
    '--output',
    '-o',
    'output_path',
    type=str,
    help='The directory where the PEtab files will be output.',
)
def models2petab(
    yaml_: str,
    output_path: str,
) -> None:
    """Create a PEtab problem for each model in a PEtab Select model YAML file.

    NB: Models in the YAML file must have a model ID.

    The output to `stdout` is a two-column tab-separated list, where the first
    column is the model ID, and the second column is the location of the PEtab
    problem YAML file for that model.

    Documentation for arguments can be viewed with
    `petab_select models2petab --help`.
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

    output_path = Path(output_path)

    for model in models:
        model_output_path = output_path / model.model_id
        model_output_path.mkdir(parents=True, exist_ok=True)
        _, petab_yaml = model.to_petab(model_output_path)
        print(f'{model.model_id}\t{petab_yaml}')


cli.add_command(candidates)
cli.add_command(model2petab)
cli.add_command(models2petab)
