"""The `Model` class."""

from __future__ import annotations

import copy
import warnings
from os.path import relpath
from typing import TYPE_CHECKING, Any, ClassVar, Literal

import mkstd
import petab.v1 as petab
from mkstd import Path
from petab.v1.C import NOMINAL_VALUE
from pydantic import (
    BaseModel,
    Field,
    PrivateAttr,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    field_serializer,
    field_validator,
    model_serializer,
    model_validator,
)

from .constants import (
    ESTIMATE,
    MODEL_HASH,
    MODEL_HASH_DELIMITER,
    MODEL_ID,
    MODEL_SUBSPACE_ID,
    MODEL_SUBSPACE_INDICES,
    MODEL_SUBSPACE_INDICES_HASH,
    MODEL_SUBSPACE_INDICES_HASH_DELIMITER,
    MODEL_SUBSPACE_INDICES_HASH_MAP,
    MODEL_SUBSPACE_PETAB_YAML,
    PETAB_PROBLEM,
    PETAB_YAML,
    ROOT_PATH,
    TYPE_PARAMETER,
    Criterion,
)
from .criteria import CriterionComputer
from .misc import (
    parameter_string_to_value,
)
from .petab import get_petab_parameters

if TYPE_CHECKING:
    from .problem import Problem

__all__ = [
    "Model",
    "default_compare",
    "ModelHash",
    "VIRTUAL_INITIAL_MODEL",
    "ModelStandard",
]


class ModelHash(BaseModel):
    """The model hash.

    The model hash is designed to be human-readable and able to be converted
    back into the corresponding model. Currently, if two models from two
    different model subspaces are actually the same PEtab problem, they will
    still have different model hashes.
    """

    model_subspace_id: str
    """The ID of the model subspace of the model.

    Unique up to a single model space.
    """
    model_subspace_indices_hash: str
    """A hash of the location of the model in its model subspace.

    Unique up to a single model subspace.
    """

    @model_validator(mode="wrap")
    def _check_kwargs(
        kwargs: dict[str, str | list[int]] | ModelHash,
        handler: ValidatorFunctionWrapHandler,
        info: ValidationInfo,
    ) -> ModelHash:
        """Handle `ModelHash` creation from different sources.

        See documentation of Pydantic wrap validators.
        """
        # Deprecate? Or introduce some `UNKNOWN_MODEL`?
        if kwargs is None:
            kwargs = VIRTUAL_INITIAL_MODEL.hash

        if isinstance(kwargs, ModelHash):
            return kwargs
        elif isinstance(kwargs, dict):
            kwargs[MODEL_SUBSPACE_INDICES_HASH] = (
                ModelHash.hash_model_subspace_indices(
                    kwargs[MODEL_SUBSPACE_INDICES]
                )
            )
            del kwargs[MODEL_SUBSPACE_INDICES]
        elif isinstance(kwargs, str):
            kwargs = ModelHash.kwargs_from_str(hash_str=kwargs)

        expected_model_hash = None
        if MODEL_HASH in kwargs:
            expected_model_hash = kwargs[MODEL_HASH]
            if isinstance(expected_model_hash, str):
                expected_model_hash = ModelHash.from_str(expected_model_hash)
            del kwargs[MODEL_HASH]

        model_hash = handler(kwargs)

        if expected_model_hash is not None:
            if model_hash != expected_model_hash:
                warnings.warn(
                    "The provided model hash is inconsistent with its model "
                    "subspace and model subspace indices. Old hash: "
                    f"`{expected_model_hash}`. New hash: `{model_hash}`.",
                    stacklevel=2,
                )

        return model_hash

    @model_serializer()
    def _serialize(self) -> str:
        return str(self)

    @staticmethod
    def kwargs_from_str(hash_str: str) -> dict[str, str]:
        """Convert a model hash string into constructor kwargs."""
        return dict(
            zip(
                [MODEL_SUBSPACE_ID, MODEL_SUBSPACE_INDICES_HASH],
                hash_str.split(MODEL_HASH_DELIMITER),
                strict=True,
            )
        )

    @staticmethod
    def hash_model_subspace_indices(model_subspace_indices: list[int]) -> str:
        """Hash the location of a model in its subspace.

        Args:
            model_subspace_indices:
                The location (indices) of the model in its subspace.

        Returns:
            The hash.
        """
        if not model_subspace_indices:
            return ""
        if max(model_subspace_indices) < len(MODEL_SUBSPACE_INDICES_HASH_MAP):
            return "".join(
                MODEL_SUBSPACE_INDICES_HASH_MAP[index]
                for index in model_subspace_indices
            )
        return MODEL_SUBSPACE_INDICES_HASH_DELIMITER.join(
            str(i) for i in model_subspace_indices
        )

    def unhash_model_subspace_indices(self) -> list[int]:
        """Get the location of a model in its subspace.

        Returns:
            The location, as indices of the subspace.
        """
        if (
            MODEL_SUBSPACE_INDICES_HASH_DELIMITER
            not in self.model_subspace_indices_hash
        ):
            return [
                MODEL_SUBSPACE_INDICES_HASH_MAP.index(s)
                for s in self.model_subspace_indices_hash
            ]
        return [
            int(s)
            for s in self.model_subspace_indices_hash.split(
                MODEL_SUBSPACE_INDICES_HASH_DELIMITER
            )
        ]

    def get_model(self, problem: Problem) -> Model:
        """Get the model that a hash corresponds to.

        Args:
            problem:
                The :class:`Problem` that will be used to look up the model.

        Returns:
            The model.
        """
        return problem.model_space.model_subspaces[
            self.model_subspace_id
        ].indices_to_model(self.unhash_model_subspace_indices())

    def __hash__(self) -> str:
        """Not the model hash! Use `Model.hash` instead."""
        return hash(str(self))

    def __eq__(self, other_hash: str | ModelHash) -> bool:
        """Check whether two model hashes are equivalent."""
        return str(self) == str(other_hash)

    def __str__(self) -> str:
        """Convert the hash to a string."""
        return MODEL_HASH_DELIMITER.join(
            [self.model_subspace_id, self.model_subspace_indices_hash]
        )

    def __repr__(self) -> str:
        """Convert the hash to a string representation."""
        return str(self)


class VirtualModelBase(BaseModel):
    """Sufficient information for the virtual initial model."""

    model_subspace_id: str
    """The ID of the subspace that this model belongs to."""
    model_subspace_indices: list[int]
    """The location of this model in its subspace."""
    criteria: dict[Criterion, float] = Field(default_factory=dict)
    """The criterion values of the calibrated model (e.g. AIC)."""
    model_hash: ModelHash = Field(default=None)
    """The model hash (treat as read-only after initialization)."""

    @model_validator(mode="after")
    def _check_hash(self: ModelBase) -> ModelBase:
        """Validate the model hash."""
        kwargs = {
            MODEL_SUBSPACE_ID: self.model_subspace_id,
            MODEL_SUBSPACE_INDICES: self.model_subspace_indices,
        }
        if self.model_hash is not None:
            kwargs[MODEL_HASH] = self.model_hash
        self.model_hash = ModelHash.model_validate(kwargs)

        return self

    @field_validator("criteria", mode="after")
    @classmethod
    def _fix_criteria_typing(
        cls, criteria: dict[str | Criterion, float]
    ) -> dict[Criterion, float]:
        """Fix criteria typing."""
        criteria = {
            (
                criterion
                if isinstance(criterion, Criterion)
                else Criterion[criterion]
            ): value
            for criterion, value in criteria.items()
        }
        return criteria

    @field_serializer("criteria")
    def _serialize_criteria(
        self, criteria: dict[Criterion, float]
    ) -> dict[str, float]:
        """Serialize criteria."""
        criteria = {
            criterion.value: value for criterion, value in criteria.items()
        }
        return criteria

    @property
    def hash(self) -> ModelHash:
        """Get the model hash."""
        return self.model_hash

    def __hash__(self) -> None:
        """Use ``Model.hash`` instead."""
        raise NotImplementedError("Use `Model.hash` instead.")

    # def __eq__(self, other_model: Model | _VirtualInitialModel) -> bool:
    #     """Check whether two model hashes are equivalent."""
    #     return self.hash == other.hash


class ModelBase(VirtualModelBase):
    """Definition of the standardized model.

    :class:`Model` is extended with additional helper methods -- use that
    instead of ``ModelBase``.
    """

    # TODO would use `FilePath` here (and remove `None` as an option),
    # but then need to handle the
    # `VIRTUAL_INITIAL_MODEL` dummy path differently.
    model_subspace_petab_yaml: Path | None
    """The location of the base PEtab problem for the model subspace.

    N.B.: Not the PEtab problem for this model specifically!
    Use :meth:`Model.to_petab` to get the model-specific PEtab
    problem.
    """
    estimated_parameters: dict[str, float] | None = Field(default=None)
    """The parameter estimates of the calibrated model (always unscaled)."""
    iteration: int | None = Field(default=None)
    """The iteration of model selection that calibrated this model."""
    model_id: str = Field(default=None)
    """The model ID."""
    model_label: str | None = Field(default=None)
    """The model label (e.g. for plotting)."""
    parameters: dict[str, float | int | Literal[ESTIMATE]]
    """PEtab problem parameters overrides for this model.

    For example, fixes parameters to certain values, or sets them to be
    estimated.
    """
    predecessor_model_hash: ModelHash = Field(default=None)
    """The predecessor model hash."""

    PATH_ATTRIBUTES: ClassVar[list[str]] = [
        MODEL_SUBSPACE_PETAB_YAML,
    ]

    @model_validator(mode="wrap")
    def _fix_relative_paths(
        data: dict[str, Any] | ModelBase,
        handler: ValidatorFunctionWrapHandler,
        info: ValidationInfo,
    ) -> ModelBase:
        if isinstance(data, ModelBase):
            return data
        model = handler(data)

        root_path = data.pop(ROOT_PATH, None)
        if root_path is None:
            return model

        model.resolve_paths(root_path=root_path)
        return model

    @model_validator(mode="after")
    def _fix_id(self: ModelBase) -> ModelBase:
        """Fix a missing ID by setting it to the hash."""
        if self.model_id is None:
            self.model_id = str(self.hash)
        return self

    @model_validator(mode="after")
    def _fix_predecessor_model_hash(self: ModelBase) -> ModelBase:
        """Fix missing predecessor model hashes.

        Sets them to ``VIRTUAL_INITIAL_MODEL.hash``.
        """
        if self.predecessor_model_hash is None:
            self.predecessor_model_hash = VIRTUAL_INITIAL_MODEL.hash
        self.predecessor_model_hash = ModelHash.model_validate(
            self.predecessor_model_hash
        )
        return self

    def to_yaml(
        self,
        filename: str | Path,
    ) -> None:
        """Save a model to a YAML file.

        All paths will be made relative to the ``filename`` directory.

        Args:
            filename:
                Location of the YAML file.
        """
        root_path = Path(filename).parent

        model = copy.deepcopy(self)
        model.set_relative_paths(root_path=root_path)
        ModelStandard.save_data(data=model, filename=filename)

    def set_relative_paths(self, root_path: str | Path) -> None:
        """Change all paths to be relative to ``root_path``."""
        root_path = Path(root_path).resolve()
        for path_attribute in self.PATH_ATTRIBUTES:
            setattr(
                self,
                path_attribute,
                relpath(
                    getattr(self, path_attribute).resolve(),
                    start=root_path,
                ),
            )

    def resolve_paths(self, root_path: str | Path) -> None:
        """Resolve all paths to be relative to ``root_path``."""
        root_path = Path(root_path).resolve()
        for path_attribute in self.PATH_ATTRIBUTES:
            setattr(
                self,
                path_attribute,
                (root_path / getattr(self, path_attribute)).resolve(),
            )


class Model(ModelBase):
    """A model."""

    # See :class:`ModelBase` for the standardized attributes. Additional
    # attributes are available in ``Model`` to improve usability.

    _model_subspace_petab_problem: petab.Problem = PrivateAttr(default=None)
    """The PEtab problem of the model subspace of this model.

    If not provided, this is reconstructed from
    :attr:`model_subspace_petab_yaml`.
    """

    @model_validator(mode="after")
    def _fix_petab_problem(self: Model) -> Model:
        """Fix a missing PEtab problem by loading it from disk."""
        if (
            self._model_subspace_petab_problem is None
            and self.model_subspace_petab_yaml is not None
        ):
            self._model_subspace_petab_problem = petab.Problem.from_yaml(
                self.model_subspace_petab_yaml
            )
        return self

    def model_post_init(self, __context: Any) -> None:
        """Add additional instance attributes."""
        self._criterion_computer = CriterionComputer(self)

    def has_criterion(self, criterion: Criterion) -> bool:
        """Check whether a value for a criterion has been set."""
        return self.criteria.get(criterion) is not None

    def set_criterion(self, criterion: Criterion, value: float) -> None:
        """Set a criterion value."""
        if self.has_criterion(criterion=criterion):
            warnings.warn(
                f"Overwriting saved criterion value. Criterion: {criterion}. "
                f"Value: `{self.get_criterion(criterion)}`.",
                stacklevel=2,
            )
        self.criteria[criterion] = float(value)

    def get_criterion(
        self,
        criterion: Criterion,
        compute: bool = True,
        raise_on_failure: bool = True,
    ) -> float | None:
        """Get a criterion value for the model.

        Args:
            criterion:
                The criterion.
            compute:
                Whether to attempt computing the criterion value. For example,
                the AIC can be computed if the likelihood is available.
            raise_on_failure:
                Whether to raise a ``ValueError`` if the criterion could not be
                computed. If ``False``, ``None`` is returned.

        Returns:
            The criterion value, or ``None`` if it is not available.
        """
        if not self.has_criterion(criterion=criterion) and compute:
            self.compute_criterion(
                criterion=criterion,
                raise_on_failure=raise_on_failure,
            )
        return self.criteria.get(criterion, None)

    def compute_criterion(
        self,
        criterion: Criterion,
        raise_on_failure: bool = True,
    ) -> float:
        """Compute a criterion value for the model.

        The value will also be stored, which will overwrite any previously
        stored value for the criterion.

        Args:
            criterion:
                The criterion.
            raise_on_failure:
                Whether to raise a ``ValueError`` if the criterion could not be
                computed. If ``False``, ``None`` is returned.

        Returns:
            The criterion value.
        """
        criterion_value = None
        try:
            criterion_value = self._criterion_computer(criterion)
            self.set_criterion(criterion, criterion_value)
        except ValueError as err:
            if raise_on_failure:
                raise ValueError(
                    "Insufficient information to compute criterion "
                    f"`{criterion}`."
                ) from err
        return criterion_value

    def set_estimated_parameters(
        self,
        estimated_parameters: dict[str, float],
        scaled: bool = False,
    ) -> None:
        """Set parameter estimates.

        Args:
            estimated_parameters:
                The estimated parameters.
            scaled:
                Whether the parameter estimates are on the scale defined in the
                PEtab problem (``True``), or unscaled (``False``).
        """
        if scaled:
            estimated_parameters = (
                self._model_subspace_petab_problem.unscale_parameters(
                    estimated_parameters
                )
            )
        self.estimated_parameters = estimated_parameters

    def to_petab(
        self,
        output_path: str | Path = None,
        set_estimated_parameters: bool | None = None,
    ) -> dict[str, petab.Problem | str | Path]:
        """Generate the PEtab problem for this model.

        Args:
            output_path:
                If specified, the PEtab tables will be written to disk, inside
                this directory.
            set_estimated_parameters:
                Whether to implement ``Model.estimated_parameters`` as the
                nominal values of the PEtab problem parameter table.
                Defaults to ``True`` if ``Model.estimated_parameters`` is set.

        Returns:
            The PEtab problem. Also returns the path of the PEtab problem YAML
            file, if ``output_path`` is specified.
        """
        petab_problem = petab.Problem.from_yaml(self.model_subspace_petab_yaml)

        if set_estimated_parameters is None and self.estimated_parameters:
            set_estimated_parameters = True

        if set_estimated_parameters:
            required_estimates = {
                parameter_id
                for parameter_id, value in self.parameters.items()
                if value == ESTIMATE
            }
            missing_estimates = required_estimates.difference(
                self.estimated_parameters
            )
            if missing_estimates:
                raise ValueError(
                    "Try again with `set_estimated_parameters=False`, because "
                    "some parameter estimates are missing. Missing estimates for: "
                    f"`{missing_estimates}`."
                )

        for parameter_id, parameter_value in self.parameters.items():
            # If the parameter is to be estimated.
            if parameter_value == ESTIMATE:
                petab_problem.parameter_df.loc[parameter_id, ESTIMATE] = 1
                if set_estimated_parameters:
                    petab_problem.parameter_df.loc[
                        parameter_id, NOMINAL_VALUE
                    ] = self.estimated_parameters[parameter_id]
            # Else the parameter is to be fixed.
            else:
                petab_problem.parameter_df.loc[parameter_id, ESTIMATE] = 0
                petab_problem.parameter_df.loc[parameter_id, NOMINAL_VALUE] = (
                    parameter_string_to_value(parameter_value)
                )

        petab_yaml = None
        if output_path is not None:
            output_path = Path(output_path)
            output_path.mkdir(exist_ok=True, parents=True)
            petab_yaml = petab_problem.to_files_generic(
                prefix_path=output_path
            )

        return {
            PETAB_PROBLEM: petab_problem,
            PETAB_YAML: petab_yaml,
        }

    def __str__(self) -> str:
        """Printable model summary."""
        parameter_ids = "\t".join(self.parameters.keys())
        parameter_values = "\t".join(str(v) for v in self.parameters.values())
        header = "\t".join(
            [MODEL_ID, MODEL_SUBSPACE_PETAB_YAML, parameter_ids]
        )
        data = "\t".join(
            [
                self.model_id,
                str(self.model_subspace_petab_yaml),
                parameter_values,
            ]
        )
        return f"{header}\n{data}"

    def __repr__(self) -> str:
        """The model hash.

        The hash can be used to reconstruct the model (see
        :meth:``ModelHash.get_model``).
        """
        return f'<petab_select.Model "{self.hash}">'

    def get_estimated_parameter_ids(self, full: bool = True) -> list[str]:
        """Get estimated parameter IDs.

        Args:
            full:
                Whether to provide all IDs, including additional parameters
                that are not part of the model selection problem but estimated.
        """
        estimated_parameter_ids = []
        if full:
            estimated_parameter_ids = (
                self._model_subspace_petab_problem.x_free_ids
            )

        # Add additional estimated parameters, and collect fixed parameters,
        # in this model's parameterization.
        fixed_parameter_ids = []
        for parameter_id, value in self.parameters.items():
            if (
                value == ESTIMATE
                and parameter_id not in estimated_parameter_ids
            ):
                estimated_parameter_ids.append(parameter_id)
            elif value != ESTIMATE:
                fixed_parameter_ids.append(parameter_id)

        # Remove fixed parameters.
        estimated_parameter_ids = [
            parameter_id
            for parameter_id in estimated_parameter_ids
            if parameter_id not in fixed_parameter_ids
        ]
        return estimated_parameter_ids

    def get_parameter_values(
        self,
        parameter_ids: list[str] | None = None,
    ) -> list[TYPE_PARAMETER]:
        """Get parameter values.

        Includes ``ESTIMATE`` for parameters that should be estimated.

        Args:
            parameter_ids:
                The IDs of parameters that values will be returned for. Order
                is maintained. Defaults to the model subspace PEtab problem
                parameters (including those not part of the model selection
                problem).

        Returns:
            The values of parameters.
        """
        nominal_values = get_petab_parameters(
            self._model_subspace_petab_problem
        )
        if parameter_ids is None:
            parameter_ids = list(nominal_values)
        return [
            self.parameters.get(parameter_id, nominal_values[parameter_id])
            for parameter_id in parameter_ids
        ]

    @staticmethod
    def from_yaml(
        filename: str | Path,
        model_subspace_petab_problem: petab.Problem | None = None,
    ) -> Model:
        """Load a model from a YAML file.

        Args:
            filename:
                The filename.
            model_subspace_petab_problem:
                A preloaded copy of the PEtab problem of the model subspace
                that this model belongs to.
        """
        model = ModelStandard.load_data(
            filename=filename,
            root_path=Path(filename).parent,
            _model_subspace_petab_problem=model_subspace_petab_problem,
        )
        return model

    def get_hash(self) -> ModelHash:
        """Deprecated. Use `Model.hash` instead."""
        warnings.warn(
            "Use `Model.hash` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.hash


def default_compare(
    model0: Model,
    model1: Model,
    criterion: Criterion,
    criterion_threshold: float = 0,
) -> bool:
    """Compare two calibrated models by their criterion values.

    It is assumed that the model ``model0`` provides a value for the criterion
    ``criterion``, or is the ``VIRTUAL_INITIAL_MODEL``.

    Args:
        model0:
            The original model.
        model1:
            The new model.
        criterion:
            The criterion.
        criterion_threshold:
            The non-negative value by which the new model must improve on the
            original model.

    Returns:
        ``True` if ``model1`` has a better criterion value than ``model0``,
        else ``False``.
    """
    if not model1.has_criterion(criterion):
        warnings.warn(
            f'Model "{model1.model_id}" does not provide a value for the '
            f'criterion "{criterion}".',
            stacklevel=2,
        )
        return False
    if model0.hash == VIRTUAL_INITIAL_MODEL_HASH or model0 is None:
        return True
    if criterion_threshold < 0:
        warnings.warn(
            "The provided criterion threshold is negative. "
            "The absolute value will be used instead.",
            stacklevel=2,
        )
        criterion_threshold = abs(criterion_threshold)
    if criterion in [
        Criterion.AIC,
        Criterion.AICC,
        Criterion.BIC,
        Criterion.NLLH,
        Criterion.SSR,
    ]:
        return (
            model1.get_criterion(criterion)
            < model0.get_criterion(criterion) - criterion_threshold
        )
    elif criterion in [
        Criterion.LH,
        Criterion.LLH,
    ]:
        return (
            model1.get_criterion(criterion)
            > model0.get_criterion(criterion) + criterion_threshold
        )
    else:
        raise NotImplementedError(f"Unknown criterion: {criterion}.")


VIRTUAL_INITIAL_MODEL = VirtualModelBase.model_validate(
    {
        "model_subspace_id": "virtual_initial_model",
        "model_subspace_indices": [],
    }
)
# TODO deprecate, use `VIRTUAL_INITIAL_MODEL.hash` instead
VIRTUAL_INITIAL_MODEL_HASH = VIRTUAL_INITIAL_MODEL.hash


ModelStandard = mkstd.YamlStandard(model=Model)
