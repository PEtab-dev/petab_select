"""The model selection problem class."""
import abc
from pathlib import Path
from typing import Union
import yaml

from .constants import (
    CRITERION,
    METHOD,
    MODEL_SPECIFICATION_FILES,
    VERSION,
)
from .model_space import ModelSpace


class Problem(abc.ABC):
    """Handle everything related to the model selection problem.

    Attributes:
        model_space:
            The model space.
        criterion:
            The criterion used to compare models.
        method:
            The method used to search the model space.
        version:
            The version of the PEtab Select format.
        yaml_path:
            The location of the selection problem YAML file. Used for relative
            paths that exist in e.g. the model specification files.
            TODO should the relative paths be relative to the YAML or the
            file that contains them?
    """
    def __init__(
        self,
        model_space: ModelSpace,
        criterion: str = None,
        method: str = None,
        version: str = None,
        yaml_path: str = None,
    ):
        self.model_space = model_space
        self.criterion = criterion
        self.method = method
        self.version = version
        self.yaml_path = yaml_path

    #def reset(self) -> None:
    #    self.model_space.reset()

    def get_path(self, relative_path: Union[str, Path]) -> Path:
        """Get the path to a resource, from a relative path.

        Args:
            relative_path:
                The path to the resource, that is relative to the PEtab Select
                problem YAML file location.

        Returns:
            The path to the resource.

        Todo:
            Unused?
        """
        if self.yaml_path is None:
            return Path(relative_path)
        return self.yaml_path.parent / relative_path

    def set_state(
        self,
        state,
    ) -> None:
        """Set the state of the problem.

        Currently, only the excluded models needs to be stored.

        Args:
            state:
                The state to restore to.
        """
        self.model_space.excluded_models = state

    def get_state(
        self,
    ):
        """Get the state of the problem.

        Currently, only the excluded models needs to be stored.

        Returns:
            The current state of the problem.
        """
        return self.model_space.excluded_models

    @staticmethod
    def from_yaml(
        yaml_path: Union[str, Path],
    ) -> 'Problem':
        """Generate a problem from a PEtab Select problem YAML file.

        Args:
            yaml_path:
                The location of the PEtab Select problem YAML file.

        Returns:
            A `Problem` instance.
        """
        yaml_path = Path(yaml_path)
        with open(yaml_path, 'r') as f:
            problem_specification = yaml.safe_load(f)

        if not problem_specification.get(MODEL_SPECIFICATION_FILES, []):
            raise KeyError(
                'The model selection problem specification file is missing '
                'model specification files.'
            )

        model_space = ModelSpace.from_files(
            [
                f
                for f in problem_specification[MODEL_SPECIFICATION_FILES]
            ],
            source_path=yaml_path.parent,
        )

        return Problem(
            model_space=model_space,
            criterion=problem_specification.get(CRITERION, None),
            method=problem_specification.get(METHOD, None),
            version=problem_specification.get(VERSION, None),
            yaml_path=yaml_path,
        )
