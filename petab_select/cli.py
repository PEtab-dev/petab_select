"""The PEtab Select command-line interface."""
import warnings
from pathlib import Path
from typing import Any, Dict, List

import click
import dill
import numpy as np
import pandas as pd
import yaml
from more_itertools import one

from . import ui
from .candidate_space import CandidateSpace, method_to_candidate_space_class
from .constants import INITIAL_MODEL_METHODS, PETAB_YAML
from .model import Model, models_from_yaml_list, models_to_yaml_list
from .problem import Problem


def read_state(filename: str) -> Dict[str, Any]:
    with open(filename, 'rb') as f:
        state = dill.load(f)

    state['problem'] = dill.loads(state['problem'])
    state['candidate_space'] = dill.loads(state['candidate_space'])

    return state


def write_state(
    state: Dict[str, Any],
    filename: str,
) -> Dict[str, Any]:
    with open(filename, 'wb') as f:
        dill.dump(state, f)


def get_state(
    problem: Problem,
    candidate_space: CandidateSpace,
) -> Dict[str, Any]:
    state = {
        'problem': dill.dumps(problem),
        'candidate_space': dill.dumps(candidate_space),
    }
    return state


def check_state_compatibility(
    state: Dict[str, Any],
    problem: Problem,
    candidate_space: CandidateSpace,
):
    # if state['problem'].method != problem.method:
    #    warnings.warn(
    #        'The method in the problem loaded from the state does not match '
    #        'the method specified in either the PEtab Select YAML file or '
    #        'specified explicitly with this command.'
    #    )
    pass


@click.group()
def cli():
    pass


@cli.command("candidates")
@click.option(
    '--problem',
    '-p',
    'problem_yaml',
    help='The PEtab Select YAML problem file.',
)
@click.option(
    '--state',
    '-s',
    'state_filename',
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
    help='The method used to identify the candidate models. Defaults to the method in the problem YAML.',
)
@click.option(
    '--previous-predecessor-model',
    '-P',
    'previous_predecessor_model_yaml',
    type=str,
    default=None,
    help='(Optional) The predecessor model used in the previous iteration of model selection.',
)
@click.option(
    '--calibrated-models',
    '-C',
    'calibrated_models_yamls',
    type=str,
    multiple=True,
    default=None,
    help='(Optional) Models that have been calibrated.',
)
@click.option(
    '--newly-calibrated-models',
    '-N',
    'newly_calibrated_models_yamls',
    type=str,
    multiple=True,
    default=None,
    help=(
        '(Optional) Models that were calibrated in the most recent iteration.'
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
    '--limit-sent',
    '-L',
    'limit_sent',
    type=float,
    default=np.inf,
    help=(
        '(Optional) Limit the number of models sent to the candidate space '
        '(which are possibly rejected and excluded from the output).'
    ),
)
@click.option(
    '--relative-paths/--absolute-paths',
    'relative_paths',
    type=bool,
    default=False,
    help='Whether to output paths relative to the output file.',
)
@click.option(
    '--excluded-models',
    '-e',
    'excluded_model_files',
    type=str,
    multiple=True,
    default=None,
    help='Exclude models in this file.',
)
@click.option(
    '--excluded-model-hashes',
    '-E',
    'excluded_model_hash_files',
    type=str,
    multiple=True,
    default=None,
    help='Exclude model hashes in this file (one model hash per line).',
)
def candidates(
    problem_yaml: str,
    state_filename: str,
    output: str,
    method: str = None,
    previous_predecessor_model_yaml: str = None,
    # best: str = None,
    calibrated_models_yamls: List[str] = None,
    newly_calibrated_models_yamls: List[str] = None,
    limit: float = np.inf,
    limit_sent: float = np.inf,
    relative_paths: bool = False,
    excluded_model_files: List[str] = None,
    excluded_model_hash_files: List[str] = None,
) -> None:
    """Search for candidate models in the model space.

    Documentation for arguments can be viewed with
    `petab_select candidates --help`.
    """
    problem = Problem.from_yaml(problem_yaml)
    if method is None:
        method = problem.method

    # `petab_select.ui.candidates` uses `petab_select.Problem.method` to generate
    # the candidate space.
    problem.method = method

    candidate_space = problem.new_candidate_space(limit=limit)

    # Setup state
    if not Path(state_filename).exists():
        Path(state_filename).parent.mkdir(parents=True, exist_ok=True)
    else:
        state = read_state(state_filename)
        check_state_compatibility(
            state=state,
            problem=problem,
            candidate_space=candidate_space,
        )
        problem = state['problem']
        candidate_space = state['candidate_space']

    excluded_models = []
    # TODO seems like default is `()`, not `None`...
    if excluded_model_files is not None:
        for model_yaml_list in excluded_model_files:
            excluded_models.extend(models_from_yaml_list(model_yaml_list))

    # TODO test
    excluded_model_hashes = []
    if excluded_model_hash_files is not None:
        for excluded_model_hash_file in excluded_model_hash_files:
            with open(excluded_model_hash_file, 'r') as f:
                excluded_model_hashes += f.read().split('\n')

    previous_predecessor_model = candidate_space.predecessor_model
    if previous_predecessor_model_yaml is not None:
        previous_predecessor_model = Model.from_yaml(
            previous_predecessor_model_yaml
        )

    # FIXME write single methods to take all models from lists of lists of
    #       models recursively
    calibrated_models = None
    if calibrated_models_yamls is not None:
        calibrated_models = {}
        for calibrated_models_yaml in calibrated_models_yamls:
            calibrated_models.update(
                {
                    model.get_hash(): model
                    for model in models_from_yaml_list(calibrated_models_yaml)
                }
            )

    newly_calibrated_models = None
    if newly_calibrated_models_yamls is not None:
        newly_calibrated_models = {}
        for newly_calibrated_models_yaml in newly_calibrated_models_yamls:
            newly_calibrated_models.update(
                {
                    model.get_hash(): model
                    for model in models_from_yaml_list(
                        newly_calibrated_models_yaml
                    )
                }
            )

    ui.candidates(
        problem=problem,
        candidate_space=candidate_space,
        previous_predecessor_model=previous_predecessor_model,
        calibrated_models=calibrated_models,
        newly_calibrated_models=newly_calibrated_models,
        limit=limit,
        limit_sent=limit_sent,
        excluded_models=excluded_models,
        excluded_model_hashes=excluded_model_hashes,
    )

    # Save state
    write_state(
        state=get_state(
            problem=problem,
            candidate_space=candidate_space,
        ),
        filename=state_filename,
    )

    # Save candidates
    models_to_yaml_list(
        models=candidate_space.models,
        output_yaml=output,
        relative_paths=relative_paths,
    )


@cli.command("model_to_petab")
@click.option(
    '--model',
    '-m',
    'models_yamls',
    multiple=True,
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
    '--model_id',
    '-i',
    'model_id',
    type=str,
    default=None,
    help=(
        '(Optional) The ID of the model to use, in case the YAML file '
        'contains multiple models.'
    ),
)
def model_to_petab(
    models_yamls: List[str],
    output_path: str,
    model_id: str = None,
) -> None:
    """Create a PEtab problem from a PEtab Select model YAML file.

    The filename for the PEtab problem YAML file is output to `stdout`.

    Documentation for arguments can be viewed with
    `petab_select model_to_petab --help`.
    """
    models = []
    for models_yaml in models_yamls:
        models.extend(models_from_yaml_list(models_yaml))

    model0 = None
    try:
        model0 = one(models)
    except:
        for model in models:
            if model.model_id == model_id:
                if model0 is not None:
                    raise ValueError('There are multiple models with this ID.')
                model0 = model
                # TODO could `break` here and remove the above `ValueError`
                #      and the `model0` logic
    if model0 is None:
        raise ValueError('Could not find a model with the specified model ID.')

    if model_id is not None and model0.model_id != model_id:
        raise ValueError(
            'The ID of the model from the YAML file does not match the '
            'specified ID.'
        )

    result = ui.model_to_petab(model0, output_path)
    print(result[PETAB_YAML])


@cli.command("models_to_petab")
@click.option(
    '--models',
    '-m',
    'models_yamls',
    type=str,
    multiple=True,
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
    models_yamls: List[str],
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
    models = []
    for models_yaml in models_yamls:
        models.extend(models_from_yaml_list(models_yaml))

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
    result_string = '\n'.join(
        [
            '\t'.join([model.model_id, result[PETAB_YAML]])
            for model, result in zip(models, results)
        ]
    )
    print(result_string)


@cli.command("best")
@click.option(
    '--problem',
    '-p',
    'problem_yaml',
    type=str,
    help='The PEtab Select YAML problem file.',
)
@click.option(
    '--models',
    '-m',
    'models_yamls',
    type=str,
    multiple=True,
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
    'state_filename',
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
    problem_yaml: str,
    models_yamls: List[str],
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

    if state_filename is not None:
        state = read_state(state_filename)
        check_state_compatibility(
            state=state,
            problem=problem,
            candidate_space=None,
        )
        set_state(
            state=state,
            problem=problem,
            candidate_space=None,
        )

    models = []
    for models_yaml in models_yamls:
        models.extend(models_from_yaml_list(models_yaml))

    best_model = ui.best(
        problem=problem,
        models=models,
        criterion=criterion,
    )
    best_model.to_yaml(output, paths_relative_to=paths_relative_to)


cli.add_command(candidates)
cli.add_command(model_to_petab)
cli.add_command(models_to_petab)
cli.add_command(best)
