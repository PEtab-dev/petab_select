"""The `ModelSpace` class and related methods."""
import abc
import itertools
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    TextIO,
    Union,
)

from more_itertools import nth
import numpy as np
import pandas as pd

from .constants import (
    Method,
    ESTIMATE,
    HEADER_ROW,
    MODEL_ID,
    MODEL_ID_COLUMN,
    MODEL_SPACE_FILE_NON_PARAMETER_COLUMNS,
    MODEL_SUBSPACE_ID,
    PARAMETER_DEFINITIONS_START,
    PARAMETER_VALUE_DELIMITER,
    PETAB_YAML,
    PETAB_YAML_COLUMN,
    TYPE_PATH,
)
from .candidate_space import CandidateSpace
from .model import Model
from .model_subspace import ModelSubspace


def read_model_space_file(filename: str) -> TextIO:
    """Read a model space file.

    The model space specification is currently expanded and written to a
    temporary file.

    Args:
        filename:
            The name of the file to be unpacked.

    Returns:
        A temporary file object, which is the unpacked file.

    Todo:
        * Consider alternatives to `_{n}` suffix for model `modelId`
        * How should the selected model be reported to the user? Remove the
          `_{n}` suffix and report the original `modelId` alongside the
          selected parameters? Generate a set of PEtab files with the
          chosen SBML file and the parameters specified in a parameter or
          condition file?
        * Don't "unpack" file if it is already in the unpacked format
        * Sort file after unpacking
        * Remove duplicates?
    """
    # FIXME rewrite to just generate models from the original file, instead of
    #       expanding all and writing to a file.
    expanded_models_file = NamedTemporaryFile(mode="r+", delete=False)
    with open(filename) as fh:
        with open(expanded_models_file.name, "w") as ms_f:
            # could replace `else` condition with ms_f.readline() here, and
            # remove `if` statement completely
            for line_index, line in enumerate(fh):
                # Skip empty/whitespace-only lines
                if not line.strip():
                    continue
                if line_index != HEADER_ROW:
                    columns = line2row(line, unpacked=False)
                    parameter_definitions = [
                        definition.split(PARAMETER_VALUE_DELIMITER)
                        for definition in columns[PARAMETER_DEFINITIONS_START:]
                    ]
                    for index, selection in enumerate(
                        itertools.product(*parameter_definitions)
                    ):
                        # TODO change MODEL_ID_COLUMN and YAML_ID_COLUMN
                        # to just MODEL_ID and YAML_FILENAME?
                        ms_f.write(
                            "\t".join(
                                [
                                    columns[MODEL_ID_COLUMN] + f"_{index}",
                                    columns[PETAB_YAML_COLUMN],
                                    *selection,
                                ]
                            )
                            + "\n"
                        )
                else:
                    ms_f.write(line)
    # FIXME replace with some 'ModelSpaceManager' object
    return expanded_models_file


def line2row(
    line: str,
    delimiter: str = "\t",
    unpacked: bool = True,
    convert_parameters_to_float: bool = True,
) -> List:
    """Parse a line from a model space file.

    Args:
        line:
            A line from a file with delimiter-separated columns.
        delimiter:
            The string that separates columns in the file.
        unpacked:
            Whether the line format is in the unpacked format. If False,
            parameter values are not converted to `float`.
        convert_parameters_to_float:
            Whether parameters should be converted to `float`.

    Returns:
        A list of column values. Parameter values are converted to `float`.
    """
    columns = line.strip().split(delimiter)
    metadata = columns[:PARAMETER_DEFINITIONS_START]
    if unpacked and convert_parameters_to_float:
        parameters = [float(p) for p in columns[PARAMETER_DEFINITIONS_START:]]
    else:
        parameters = columns[PARAMETER_DEFINITIONS_START:]
    return metadata + parameters


class ModelSpace():
    """A model space, as a collection of model subspaces.

    Attributes:
        model_subspaces:
            List of model subspaces.
        exclusions:
            Hashes of models that are excluded from the model space.
    """
    def __init__(
        self,
        model_subspaces: List[ModelSubspace],
    ):
        self.model_subspaces = {
            model_subspace.model_subspace_id: model_subspace
            for model_subspace in model_subspaces
        }
        self.exclusions = []

    @staticmethod
    def from_files(
        filenames: List[TYPE_PATH],
    ):
        """Create a model space from model space files.

        Args:
            filenames:
                The locations of the model space files.

        Returns:
            The corresponding model space.
        """
        # TODO validate input?
        model_space_dfs = [get_model_space_df(filename) for filename in filenames]
        model_subspaces = []
        for model_space_df, model_space_filename in zip(model_space_dfs, filenames):
            for index, definition in model_space_df.iterrows():
                model_subspaces.append(ModelSubspace.from_definition(
                    definition=definition,
                    parent_path=Path(model_space_filename).parent
                ))
        model_space = ModelSpace(model_subspaces=model_subspaces)
        return model_space

    def search(
        self,
        candidate_space: CandidateSpace,
        limit: int = np.inf,
        exclude: bool = True,
    ):
        """...TODO

        Args:
            limit:
                Note that using a limit may produce unexpected results. For
                example, it may bias candidate models to be chosen only from
                a subset of model subspaces.
        """
        # TODO change dict to list of subspaces. Each subspace should manage its own
        #      ID
        for model_subspace in self.model_subspaces.values():
            model_subspace.search(
                candidate_space=candidate_space,
                limit=limit,
                exclude=exclude,
            )
            if len(candidate_space.models) == limit:
                break
            elif len(candidate_space.models) > limit:
                raise ValueError(
                    'An unknown error has occurred. Too many models were '
                    f'generated. Requested limit: {limit}. Number of '
                    f'generated models: {len(candidate_space.models)}.'
                )

        ## FIXME implement source_path.. somewhere
        #if self.source_path is not None:
        #    for model in candidate_space.models:
        #        # TODO do this change elsewhere instead?
        #        # e.g. model subspace
        #        model.petab_yaml = self.source_path / model.petab_yaml

        if exclude:
            self.exclude_models(candidate_space.models)

        return candidate_space.models

    def __len__(self):
        """Get the number of models in this space."""
        subspace_coumts = [len(s) for s in self.model_subspaces]
        total_count = sum(subspace_counts)
        return total_count

    def exclude_model(self, model: Model):
        # FIXME add Exclusions Mixin (or object) to handle exclusions on the subspace
        # and space level.
        for model_subspace in self.model_subspaces.values():
            model_subspace.exclude_model(model)

    def exclude_models(self, models: Iterable[Model]):
        # FIXME add Exclusions Mixin (or object) to handle exclusions on the subspace
        # and space level.

        for model_subspace in self.model_subspaces.values():
            model_subspace.exclude_models(models)
            #model_subspace.reset_exclusions()

    def reset_exclusions(
        self,
        exclusions: Optional[Union[List[Any], None]] = None,
    ) -> None:
        """Reset the exclusions in the model subspaces."""
        for model_subspace in self.model_subspaces.values():
            model_subspace.reset_exclusions(exclusions)


def get_model_space_df(filename: TYPE_PATH) -> pd.DataFrame:
    #model_space_df = pd.read_csv(filename, sep='\t', index_col=MODEL_SUBSPACE_ID)  # FIXME
    model_space_df = pd.read_csv(filename, sep='\t')
    return model_space_df


def get_model_space(
    filename: TYPE_PATH,
) -> List[ModelSubspace]:
    model_space_df = get_model_space_df(filename)
    model_subspaces = []
    for definition in model_space_df.iterrows():
        model_subspaces.append(ModelSubspace.from_definition(definition))
    model_space = ModelSpace(model_subspaces=model_subspaces)
    return model_space
