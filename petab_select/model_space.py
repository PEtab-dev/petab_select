"""The `ModelSpace` class and related methods."""
import abc
import itertools
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import (
    Callable,
    Iterable,
    List,
    Optional,
    TextIO,
    Union,
)

from more_itertools import nth

from .constants import (
    ESTIMATE_SYMBOL_INTERNAL,
    ESTIMATE_SYMBOL_UI,
    HEADER_ROW,
    MODEL_ID,
    MODEL_ID_COLUMN,
    MODEL_SPACE_SPECIFICATION_NOT_PARAMETERS,
    PARAMETER_DEFINITIONS_START,
    PARAMETER_VALUE_DELIMITER,
    PETAB_YAML,
    PETAB_YAML_COLUMN,
)
from .candidate_space import CandidateSpace
from .model import Model


def read_model_specification_file(filename: str) -> TextIO:
    """Read a model specification file.

    The specification is currently expanded and written to a temporary file.

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
                        _replace_estimate_symbol(
                            definition.split(PARAMETER_VALUE_DELIMITER)
                        )
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
    """Parse a line from a model specifications file.

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


def _replace_estimate_symbol(parameter_definition: List[str]) -> List:
    """
    Converts the user-friendly symbol for estimated parameters, to the internal
    symbol.

    Args:
        parameter_definition:
            A definition for a single parameter from a row of the model
            specification file. The definition should be split into a list by
            PARAMETER_VALUE_DELIMITER.

    Returns:
        The parameter definition, with the user-friendly estimate symbol
        substituted for the internal symbol.
    """
    return [
        ESTIMATE_SYMBOL_INTERNAL if p == ESTIMATE_SYMBOL_UI else p
        for p in parameter_definition
    ]


def model_generator_from_file(
    file_: TextIO,
) -> Callable:
    """Get a generator for models described by the model specification file.

    Args:
        exclude_history:
            If `True`, models with Id's in `self.selection_history` are not
            yielded.
        exclusions:
            A list of model Id's to avoid yielding.

    Yields:
        The next model, as a dictionary, where the keys are the column headers
        in the model specification file, and the values are the respective
        column values in a row of the model specification file.
    """
    def generator() -> Iterable[Model]:
        # Go to the start of model specification rows.
        file_.seek(0)
        # First line is the header
        header = line2row(
            file_.readline(),
            convert_parameters_to_float=False,
        )

        # TODO efficient exclusions at this stage?
        # if exclusions is None:
        #     exclusions = []
        # if exclude_history:
        #     exclusions += self.selection_history.keys()

        for model_index, line in enumerate(file_):
            model_dict = dict(zip(header, line2row(line)))
            model = Model(
                model_id=model_dict[MODEL_ID],
                petab_yaml=model_dict[PETAB_YAML],
                parameters={
                    **{
                        k: v
                        for k, v in model_dict.items()
                        if k not in MODEL_SPACE_SPECIFICATION_NOT_PARAMETERS
                    },
                },
                index=model_index,
            )

            # # Exclusion of history makes sense here, to avoid duplicated code
            # # in specific selectors. However, the selection history of this
            # # class is only updated when a `selector.__call__` returns, so
            # # the exclusion of a model tested twice within the same selector
            # # call is not excluded here. Could be implemented by decorating
            # # `model_generator` in `ModelSelectorMethod` to include the
            # # call selection history as `exclusions` (TODO).
            # if model_dict[MODEL_ID] in exclusions:
            #     continue

            yield model
    return generator


class ModelSpace(abc.ABC):
    """The model space.

    Attributes:
        excluded_models:
            Hashes of models to be excluded when generating candidate models
            during a search of the model space.
        generator:
            A generator to generate an iterator over the model space.
        parameter_ids:
            The ordered list of parameter IDs, from the columns of the model
            specification file.
        source_path:
            The path that the location of PEtab problem YAML files, as
            specified in model specification files, is relative to.

    Todo:
        Remove dependence on `parameter_ids`.
    """
    def __init__(
        self,
        generator: Callable[[], Iterable[Model]],
        source_path: Optional[Union[str, Path]] = None,
        excluded_models: Optional[List[int]] = None,
    ):
        self.generator = generator
        self.source_path = None
        if source_path is not None:
            self.source_path = Path(source_path)
        if excluded_models is None:
            excluded_models = []

        self.reset(excluded_models=excluded_models)

        # FIXME currently just uses the number of estimated parameters as
        #       defined in the model spec file. However, the PEtab parameters
        #       table also has other estimated parameters. Also, there may be
        #       multiple model spec files, each with different estimated
        #       parameters. Hence, the PEtab parameters table should be used
        #       to determine the full vector of estimated parameters.
        #       1. Generate all PEtab problems according to the model spec
        #          file(s).
        #       2. Construct the superset of all estimated parameters across
        #          all problems.
        #       3. Raise warning for missing parameters between different
        #          models?
        #       4. Store the superset as a list of parameter IDs, instead of
        #          the current, ambiguous, `max_estimated` (only correct for
        #          a single model spec file with only a single PEtab
        #          problem...)
        #       5. If a model is missing a parameter, assume it is "fixed" to
        #          0?
        space = generator()
        self.parameter_ids = list(next(space).parameters.keys())

    def neighbors(
        self,
        candidate_space: CandidateSpace,
        limit: int = None,
        exclude: bool = True,
    ) -> List[Model]:
        """Find neighbors of the model in the model space.

        Args:
            candidate_space:
                A handler for model candidates.
            limit:
                Limit the number of returned models to this value.
            exclude:
                Whether to exclude neighbors from the model space.

        Returns:
            The candidate models.
        """
        if limit is None:
            limit = -1

        space = self.generator()

        for model in space:
            # TODO use hash?
            # if hash(model) in self.excluded_models:
            if model.index in self.excluded_models:
                continue
            candidate_space.consider(model)
            if len(candidate_space.models) == limit:
                break

        if self.source_path is not None:
            for model in candidate_space.models:
                # TODO do this change elsewhere instead?
                model.petab_yaml = self.source_path / model.petab_yaml

        if exclude:
            self.exclude_models(candidate_space.models)

        return candidate_space.models

    def index(self, index: int) -> Optional[Model]:
        """Get a model in the model space, by an index.

        Args:
            index:
                The index of the model in the model space.

        Returns:
            The model, or `None` if the index doesn't match a model.
        """
        return nth(self.generator(), index, None)

    def exclude_models(
        self,
        excluded_models: Iterable[Model],
    ) -> None:
        """Exclude models from the model space.

        These models will be skipped in the `ModelSpace.neighbors` method.

        Args:
            excluded_models:
                The models to exclude.
        """
        for model in excluded_models:
            self.excluded_models.append(model.index)

    def reset(
        self,
        excluded_models: Optional[List[int]] = None,
    ) -> None:
        """Reset the excluded models.

        Args:
            excluded_models:
                Models to exclude from the model space.
        """
        if excluded_models is None:
            excluded_models = []
        self.excluded_models = excluded_models

    @staticmethod
    def from_file(
        filename: str,
    ) -> 'ModelSpace':
        """Generate a model space from a model specifications file.

        Args:
            filename:
                The location of the model specifications file.

        Returns:
            The model space.
        """
        return ModelSpace(
            generator=model_generator_from_file(
                read_model_specification_file(filename)
            ),
        )

    @staticmethod
    def from_files(
        filenames: Iterable[str],
        alternate: bool = True,
        source_path: Union[str, Path] = None,
    ) -> 'ModelSpace':
        """Generate a model space from model specifications files.

        Args:
            filenames:
                The locations of the model specifications files.
            alternate:
                Whether generation of models from the files should exhaust
                models from one file before the next, or alternate between
                files.
            source_path:
                The path that the location of files, as specified in model
                specification files, is relative to.

        Returns:
            The model space.
        """
        if source_path is not None:
            source_path = Path(source_path)
        generators = [
            model_generator_from_file(
                read_model_specification_file(
                    (source_path / filename)
                    if source_path is not None
                    else filename,
                )
            )
            for filename in filenames
        ]

        def generator():
            exhausted = [False for _ in generators]
            spaces = [g() for g in generators]
            while True:
                for space_index, space in enumerate(spaces):
                    # Loop over all model spaces, generating one model from
                    # each file until all files are exhausted.
                    if alternate:
                        if not exhausted[space_index]:
                            try:
                                model = next(space)
                                yield model
                            except StopIteration:
                                exhausted[space_index] = True
                                if all(exhausted):
                                    return
                    # Loop over all model spaces, generating all models from
                    # one file before generating models from the next file.
                    else:
                        for model in space:
                            yield model
                        exhausted[space_index] = True
                        if all(exhausted):
                            raise StopIteration

        return ModelSpace(
            generator=generator,
            source_path=source_path,
        )
