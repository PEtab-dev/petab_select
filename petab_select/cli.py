"""The PEtab Select command-line interface."""
from pathlib import Path
import yaml

import click
import dill

from .candidate_space import (
    BackwardCandidateSpace,
    BruteForceCandidateSpace,
    CandidateSpace,
    ForwardCandidateSpace,
    LateralCandidateSpace,
    INITIAL_MODEL_METHODS,
)
from .model import Model
from .problem import Problem


@click.group()
def cli():
    pass


def parse_candidate_space(
    method: str,
    initial_model_yaml_path: str,
) -> CandidateSpace:
    """Generate an appropriate candidate space instance.

    Args:
        method:
            The method used to identify candidate models.
        initial_model_yaml_path:
            The location of a PEtab Select model YAML file, that will be used
            to initialize a search for candidates if applicable.

    Returns:
        An instance of a CandidateSpace subclass.
    """
    if method in INITIAL_MODEL_METHODS and initial_model_yaml_path is None:
        raise ValueError(
            'Please supply the initial model when using forward or backward '
            'selection.'
        )

    model0 = None
    if initial_model_yaml_path is not None:
        model0 = Model.from_yaml(initial_model_yaml_path)

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
    'state',
    '-s',
    type=str,
    help='The file that stores the state.',
)
@click.option(
    'output',
    '-o',
    type=str,
    help='The file where candidate models will be stored.',
)
@click.option(
    'method',
    '-m',
    type=str,
    help='The method used to identify the candidate models.',
)
@click.option(
    'initial',
    '-i',
    type=str,
    default=None,
    help='(Optional) The initial model used in the candidate model search.',
)
@click.option(
    'limit',
    '-l',
    type=int,
    default=None,
    help='(Optional) Limit the number of models in the output.',
)
def search(
    yaml_: str,
    state: str,
    output: str,
    method: str,
    initial: str = None,
    limit: int = None,
):
    """Search for candidate models in the model space.

    Documentation for arguments can be viewed with
    `petab_select search --help`.
    """
    candidate_space = parse_candidate_space(
        method=method,
        initial_model_yaml_path=initial,
    )

    problem = Problem.from_yaml(yaml_)
    if Path(state).exists():
        # Load state
        with open(state, 'rb') as f:
            problem.set_state(dill.load(f))
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


cli.add_command(search)
