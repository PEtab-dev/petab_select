"""The model selection problem class."""

from collections.abc import Callable, Iterable
from functools import partial
from pathlib import Path
from typing import Any

import yaml

from .candidate_space import CandidateSpace, method_to_candidate_space_class
from .constants import (
    CANDIDATE_SPACE_ARGUMENTS,
    CRITERION,
    METHOD,
    MODEL_SPACE_FILES,
    PREDECESSOR_MODEL,
    PROBLEM_ID,
    VERSION,
    Criterion,
    Method,
)
from .model import Model, ModelHash, default_compare
from .model_space import ModelSpace

__all__ = [
    "Problem",
]


class Problem:
    """Handle everything related to the model selection problem.

    Attributes:
        model_space:
            The model space.
        calibrated_models:
            Calibrated models. Will be used to augment the model selection problem (e.g.
            by excluding them from the model space).
        candidate_space_arguments:
            Arguments are forwarded to the candidate space constructor.
        compare:
            A method that compares models by selection criterion. See
            :func:`petab_select.model.default_compare` for an example.
        criterion:
            The criterion used to compare models.
        method:
            The method used to search the model space.
        version:
            The version of the PEtab Select format.
        yaml_path:
            The location of the selection problem YAML file. Used for relative
            paths that exist in e.g. the model space files.
    """

    """
    FIXME(dilpath)
    Unsaved attributes:
        candidate_space:
            The candidate space that will be used.
            Reason for not saving:
                Essentially reproducible from :attr:`Problem.method` and
                :attr:`Problem.calibrated_models`.
    FIXME(dilpath) refactor calibrated_models out, move to e.g. candidate
                   space args
            TODO should the relative paths be relative to the YAML or the file that contains them?
                 problem relative to file that contains them

    """

    def __init__(
        self,
        model_space: ModelSpace,
        candidate_space_arguments: dict[str, Any] = None,
        compare: Callable[[Model, Model], bool] = None,
        criterion: Criterion = None,
        problem_id: str = None,
        method: str = None,
        version: str = None,
        yaml_path: Path | str = None,
    ):
        self.model_space = model_space
        self.criterion = criterion
        self.problem_id = problem_id
        self.method = method
        self.version = version
        self.yaml_path = Path(yaml_path)

        self.candidate_space_arguments = candidate_space_arguments
        if self.candidate_space_arguments is None:
            self.candidate_space_arguments = {}

        self.compare = compare
        if self.compare is None:
            self.compare = partial(default_compare, criterion=self.criterion)

    def __str__(self):
        return (
            f"YAML: {self.yaml_path}\n"
            f"Method: {self.method}\n"
            f"Criterion: {self.criterion}\n"
            f"Version: {self.version}\n"
        )

    def get_path(self, relative_path: str | Path) -> Path:
        """Get the path to a resource, from a relative path.

        Args:
            relative_path:
                The path to the resource, that is relative to the PEtab Select
                problem YAML file location.

        Returns:
            The path to the resource.
        """
        """
        TODO:
            Unused?
        """
        if self.yaml_path is None:
            return Path(relative_path)
        return self.yaml_path.parent / relative_path

    def exclude_models(
        self,
        models: Iterable[Model],
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
        self.model_space.exclude_model_hashes(model_hashes)

    @staticmethod
    def from_yaml(
        yaml_path: str | Path,
    ) -> "Problem":
        """Generate a problem from a PEtab Select problem YAML file.

        Args:
            yaml_path:
                The location of the PEtab Select problem YAML file.

        Returns:
            A `Problem` instance.
        """
        yaml_path = Path(yaml_path)
        with open(yaml_path) as f:
            problem_specification = yaml.safe_load(f)

        if not problem_specification.get(MODEL_SPACE_FILES, []):
            raise KeyError(
                "The model selection problem specification file is missing "
                "model space files."
            )

        model_space = ModelSpace.from_files(
            # problem_specification[MODEL_SPACE_FILES],
            [
                # `pathlib.Path` appears to handle absolute `model_space_file` paths
                # correctly, even if used as a relative path.
                # TODO test
                # This is similar to the `Problem.get_path` method.
                yaml_path.parent / model_space_file
                for model_space_file in problem_specification[
                    MODEL_SPACE_FILES
                ]
            ],
            # source_path=yaml_path.parent,
        )

        criterion = problem_specification.get(CRITERION, None)
        if criterion is not None:
            criterion = Criterion(criterion)

        problem_id = problem_specification.get(PROBLEM_ID, None)

        candidate_space_arguments = problem_specification.get(
            CANDIDATE_SPACE_ARGUMENTS,
            None,
        )
        if candidate_space_arguments is not None:
            if PREDECESSOR_MODEL in candidate_space_arguments:
                candidate_space_arguments[PREDECESSOR_MODEL] = (
                    yaml_path.parent
                    / candidate_space_arguments[PREDECESSOR_MODEL]
                )

        return Problem(
            model_space=model_space,
            candidate_space_arguments=candidate_space_arguments,
            criterion=criterion,
            # TODO refactor method to use enum
            method=problem_specification.get(METHOD, None),
            problem_id=problem_id,
            version=problem_specification.get(VERSION, None),
            yaml_path=yaml_path,
        )

    def get_best(
        self,
        models: list[Model] | dict[ModelHash, Model] | None,
        criterion: str | None | None = None,
        compute_criterion: bool = False,
    ) -> Model:
        """Get the best model from a collection of models.

        The best model is selected based on the selection problem's criterion.

        Args:
            models:
                The best model will be taken from these models.
            criterion:
                The criterion by which models will be compared. Defaults to
                ``self.criterion`` (e.g. as defined in the PEtab Select problem YAML
                file).
            compute_criterion:
                Whether to try computing criterion values, if sufficient
                information is available (e.g., likelihood and number of
                parameters, to compute AIC).

        Returns:
            The best model.
        """
        if isinstance(models, dict):
            models = list(models.values())
        if criterion is None:
            criterion = self.criterion

        best_model = None
        for model in models:
            if compute_criterion and not model.has_criterion(criterion):
                model.get_criterion(criterion)
            if best_model is None:
                if model.has_criterion(criterion):
                    best_model = model
                # TODO warn if criterion is not available?
                continue
            if self.compare(best_model, model, criterion=criterion):
                best_model = model
        if best_model is None:
            raise KeyError(
                f"None of the supplied models have a value set for the criterion {criterion}."
            )
        return best_model

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
