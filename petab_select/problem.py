"""The model selection problem class."""

from __future__ import annotations

import copy
import warnings
from collections.abc import Callable, Iterable
from functools import partial
from os.path import relpath
from pathlib import Path
from typing import Annotated, Any

import mkstd
from pydantic import (
    BaseModel,
    Field,
    PlainSerializer,
    PrivateAttr,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    model_validator,
)

from .analyze import get_best
from .candidate_space import CandidateSpace, method_to_candidate_space_class
from .constants import (
    CRITERION,
    PREDECESSOR_MODEL,
    ROOT_PATH,
    TYPE_PATH,
    Criterion,
    Method,
)
from .model import Model, ModelHash, default_compare
from .model_space import ModelSpace
from .models import Models

__all__ = [
    "Problem",
    "ProblemStandard",
]


class State(BaseModel):
    """Carry the state of applying model selection methods to the problem."""

    models: Models = Field(default_factory=Models)
    """All calibrated models."""
    iteration: int = Field(default=0)
    """The latest iteration of model selection."""

    def increment_iteration(self) -> None:
        """Start the next iteration."""
        self.iteration += 1

    def reset(self) -> None:
        """Reset the state.

        N.B.: does not reset all state information, which currently also exists
        in other classes. Open a GitHub issue if you see unusual behavior. A
        quick fix is to simply recreate the PEtab Select problem, and any other
        objects that you use, e.g. the candidate space, whenever you need a
        full reset.
        https://github.com/PEtab-dev/petab_select/issues
        """
        # FIXME state information is currently distributed across multiple
        # classes, e.g. exclusions in model subspaces and candidate spaces.
        # move all state information here.
        self.models = Models()
        self.iteration = 0


class Problem(BaseModel):
    """The model selection problem."""

    format_version: str = Field(default="1.0.0")
    """The file format version."""
    criterion: Annotated[
        Criterion, PlainSerializer(lambda x: x.value, return_type=str)
    ]
    """The criterion used to compare models."""
    method: Annotated[
        Method, PlainSerializer(lambda x: x.value, return_type=str)
    ]
    """The method used to search the model space."""
    model_space_files: list[Path]
    """The files that define the model space."""
    candidate_space_arguments: dict[str, Any] = Field(default_factory=dict)
    """Method-specific arguments.

    These are forwarded to the candidate space constructor.
    """

    _compare: Callable[[Model, Model], bool] = PrivateAttr(default=None)
    """The method by which models are compared."""
    _state: State = PrivateAttr(default_factory=State)

    @model_validator(mode="wrap")
    def _check_input(
        data: dict[str, Any] | Problem,
        handler: ValidatorFunctionWrapHandler,
        info: ValidationInfo,
    ) -> Problem:
        if isinstance(data, Problem):
            return data

        compare = data.pop("compare", None) or data.pop("_compare", None)
        if "state" in data:
            data["_state"] = data["state"]
        root_path = Path(data.pop(ROOT_PATH, ""))

        problem = handler(data)

        if compare is None:
            compare = partial(default_compare, criterion=problem.criterion)
        problem._compare = compare

        problem._model_space = ModelSpace.load(
            [
                root_path / model_space_file
                for model_space_file in problem.model_space_files
            ]
        )

        if PREDECESSOR_MODEL in problem.candidate_space_arguments:
            problem.candidate_space_arguments[PREDECESSOR_MODEL] = (
                root_path
                / problem.candidate_space_arguments[PREDECESSOR_MODEL]
            )

        return problem

    @property
    def state(self) -> State:
        return self._state

    @staticmethod
    def from_yaml(filename: TYPE_PATH) -> Problem:
        """Load a problem from a YAML file."""
        problem = ProblemStandard.load_data(
            filename=filename,
            root_path=Path(filename).parent,
        )
        return problem

    def to_yaml(
        self,
        filename: str | Path,
    ) -> None:
        """Save a problem to a YAML file.

        All paths will be made relative to the ``filename`` directory.

        Args:
            filename:
                Location of the YAML file.
        """
        root_path = Path(filename).parent

        problem = copy.deepcopy(self)
        problem.model_space_files = [
            relpath(
                model_space_file.resolve(),
                start=root_path,
            )
            for model_space_file in problem.model_space_files
        ]
        ProblemStandard.save_data(data=problem, filename=filename)

    def save(
        self,
        directory: str | Path,
    ) -> None:
        """Save all data (problem and model space) to a ``directory``.

        Inside the directory, two files will be created:
        (1) ``petab_select_problem.yaml``, and
        (2) ``model_space.tsv``.

        All paths will be made relative to the ``directory``.
        """
        directory = Path(directory)
        directory.mkdir(exist_ok=True, parents=True)

        problem = copy.deepcopy(self)
        problem.model_space_files = ["model_space.tsv"]
        if PREDECESSOR_MODEL in problem.candidate_space_arguments:
            problem.candidate_space_arguments[PREDECESSOR_MODEL] = relpath(
                problem.candidate_space_arguments[PREDECESSOR_MODEL],
                start=directory,
            )
        ProblemStandard.save_data(
            data=problem, filename=directory / "petab_select_problem.yaml"
        )

        problem.model_space.save(filename=directory / "model_space.tsv")

    @property
    def compare(self):
        return self._compare

    @property
    def model_space(self):
        return self._model_space

    def __str__(self):
        return (
            f"Method: {self.method}\n"
            f"Criterion: {self.criterion}\n"
            f"Format version: {self.format_version}\n"
        )

    def exclude_models(
        self,
        models: Models,
    ) -> None:
        """Exclude models from the model space.

        Args:
            models:
                The models.
        """
        self.model_space.exclude_models(models)

    def exclude_model_hashes(
        self,
        model_hashes: Iterable[str],
    ) -> None:
        """Exclude models from the model space, by model hashes.

        Args:
            model_hashes:
                The model hashes.
        """
        # FIXME think about design here -- should we have exclude_models here?
        warnings.warn(
            "Use `exclude_models` instead. It also accepts hashes.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.exclude_models(models=Models(models=model_hashes, problem=self))

    def get_best(
        self,
        models: Models,
        # models: list[Model] | dict[ModelHash, Model] | None,
        criterion: str | None | None = None,
        compute_criterion: bool = False,
    ) -> Model:
        """Get the best model from a collection of models.

        The best model is selected based on the selection problem's criterion.

        Args:
            models:
                The models.
            criterion:
                The criterion. Defaults to the problem criterion.
            compute_criterion:
                Whether to try computing criterion values, if sufficient
                information is available (e.g., likelihood and number of
                parameters, to compute AIC).

        Returns:
            The best model.
        """
        warnings.warn(
            "Use ``petab_select.ui.get_best`` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if criterion is None:
            criterion = self.criterion

        return get_best(
            models=models,
            criterion=criterion,
            compare=self.compare,
            compute_criterion=compute_criterion,
        )

    def model_hash_to_model(self, model_hash: str | ModelHash) -> Model:
        """Get the model that matches a model hash.

        Args:
            model_hash:
                The model hash.

        Returns:
            The model.
        """
        return ModelHash.from_hash(model_hash).get_model(
            petab_select_problem=self,
        )

    def get_model(
        self, model_subspace_id: str, model_subspace_indices: list[int]
    ) -> Model:
        return self.model_space.model_subspaces[
            model_subspace_id
        ].indices_to_model(model_subspace_indices)

    def new_candidate_space(
        self,
        *args,
        method: Method = None,
        **kwargs,
    ) -> CandidateSpace:
        """Construct a new candidate space.

        Args:
            args, kwargs:
                Arguments are passed to the candidate space constructor.
            method:
                The model selection method.
        """
        if method is None:
            method = self.method
        kwargs[CRITERION] = kwargs.get(CRITERION, self.criterion)
        candidate_space_class = method_to_candidate_space_class(method)
        candidate_space_arguments = (
            candidate_space_class.read_arguments_from_yaml_dict(
                self.candidate_space_arguments
            )
        )
        candidate_space_kwargs = {
            **candidate_space_arguments,
            **kwargs,
        }
        candidate_space = candidate_space_class(
            *args,
            **candidate_space_kwargs,
        )
        return candidate_space


ProblemStandard = mkstd.YamlStandard(model=Problem)
