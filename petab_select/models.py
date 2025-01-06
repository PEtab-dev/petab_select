from __future__ import annotations

import copy
import warnings
from collections import Counter
from collections.abc import Iterable, MutableSequence
from pathlib import Path
from typing import TYPE_CHECKING, Any, TypeAlias

import mkstd
import numpy as np
import pandas as pd
from pydantic import (
    Field,
    PrivateAttr,
    RootModel,
    ValidationInfo,
    ValidatorFunctionWrapHandler,
    model_validator,
)

from .constants import (
    CRITERIA,
    ESTIMATED_PARAMETERS,
    ITERATION,
    MODEL_HASH,
    MODEL_ID,
    MODEL_SUBSPACE_PETAB_PROBLEM,
    PREDECESSOR_MODEL_HASH,
    ROOT_PATH,
    TYPE_PATH,
    Criterion,
)
from .model import (
    Model,
    ModelHash,
    VirtualModelBase,
)

if TYPE_CHECKING:
    import petab

    from .problem import Problem

    # `Models` can be constructed from actual `Model`s,
    # or `ModelHash`s, or the `str` of a model hash.
    ModelLike: TypeAlias = Model | ModelHash | str
    ModelsLike: TypeAlias = "Models" | Iterable[Model | ModelHash | str]
    # Access a model by list index, model hash, slice of indices, model hash
    # string, or an iterable of these things.
    ModelIndex: TypeAlias = int | ModelHash | slice | str | Iterable

__all__ = [
    "_ListDict",
    "Models",
    "models_from_yaml_list",
    "models_to_yaml_list",
    "ModelsStandard",
]


class _ListDict(RootModel, MutableSequence):
    """Acts like a ``list`` and a ``dict``.

    Not all methods are implemented -- feel free to request anything that you
    think makes sense for a ``list`` or ``dict`` object.

    The context is a list of objects that may have some metadata (e.g. a hash)
    associated with each of them. The objects can be operated on like a list,
    or requested like a dict, by their metadata (hash).

    Mostly based on ``UserList`` and ``UserDict``, but some methods are
    currently not yet implemented.
    https://github.com/python/cpython/blob/main/Lib/collections/__init__.py

    The typing is currently based on PEtab Select objects. Hence, objects are
    in ``_models``, and metadata (model hashes) are in ``_hashes``.
    """

    root: list[Model] = Field(default_factory=list)
    """The list of models."""
    _hashes: list[ModelHash] = PrivateAttr(default_factory=list)
    """The list of model hashes."""
    _problem: Problem | None = PrivateAttr(default=None)
    """The PEtab Select problem that all models belong to.

    If this is provided, then you can add models by hashes.
    """

    @model_validator(mode="wrap")
    def _check_kwargs(
        kwargs: dict[str, list[ModelLike] | Problem] | list[ModelLike],
        handler: ValidatorFunctionWrapHandler,
        info: ValidationInfo,
    ) -> Models:
        """Handle `Models` creation from different sources."""
        _models = []
        _problem = None
        if isinstance(kwargs, list):
            _models = kwargs
        elif isinstance(kwargs, dict):
            # Identify the models
            if "models" in kwargs and "root" in kwargs:
                raise ValueError("Provide only one of `root` and `models`.")
            _models = kwargs.get("models") or kwargs.get("root") or []

            # Identify the PEtab Select problem
            if "problem" in kwargs and "_problem" in kwargs:
                raise ValueError(
                    "Provide only one of `problem` and `_problem`."
                )
            _problem = kwargs.get("problem") or kwargs.get("_problem")

            # Distribute model constructor kwargs to each model dict
            if model_kwargs := kwargs.get("model_kwargs"):
                for _model_index, _model in enumerate(_models):
                    if not isinstance(_model, dict):
                        raise ValueError(
                            "`model_kwargs` are only intended to be used when "
                            "constructing models from a YAML file."
                        )
                    _models[_model_index] = {**_model, **model_kwargs}

        models = handler(_models)
        models._problem = _problem
        return models

    @model_validator(mode="after")
    def _check_typing(self: RootModel) -> RootModel:
        """Fix model typing."""
        models0 = self._models
        self.root = []
        # This also converts all model hashes into models.
        self.extend(models0)
        return self

    @property
    def _models(self) -> list[Model]:
        return self.root

    def __repr__(self) -> str:
        """Get the model hashes that can regenerate these models.

        N.B.: some information, e.g. criteria, will be lost if the hashes are
        used to reproduce the set of models.
        """
        return repr(self._hashes)

    # skipped __lt__, __le__

    def __eq__(self, other) -> bool:
        other_hashes = Models(models=other)._hashes
        same_length = len(self._hashes) == len(other_hashes)
        same_hashes = set(self._hashes) == set(other_hashes)
        return same_length and same_hashes

    # skipped __gt__, __ge__, __cast

    def __contains__(self, item: ModelLike) -> bool:
        match item:
            case Model():
                return item in self._models
            case ModelHash() | str():
                return item in self._hashes
            case VirtualModelBase():
                return False
            case _:
                raise TypeError(f"Unexpected type: `{type(item)}`.")

    def __len__(self) -> int:
        return len(self._models)

    def __getitem__(
        self, key: ModelIndex | Iterable[ModelIndex]
    ) -> Model | Models:
        try:
            match key:
                case int():
                    return self._models[key]
                case ModelHash() | str():
                    return self._models[self._hashes.index(key)]
                case slice():
                    return self.__class__(self._models[key])
                case Iterable():
                    # TODO sensible to yield here?
                    return [self[key_] for key_ in key]
                case _:
                    raise TypeError(f"Unexpected type: `{type(key)}`.")
        except ValueError as err:
            raise KeyError from err

    def _model_like_to_model(self, model_like: ModelLike) -> Model:
        """Get the model that corresponds to a model-like object.

        Args:
            model_like:
                Something that uniquely identifies a model; a model or a model
                hash.

        Returns:
            The model.
        """
        match model_like:
            case Model():
                model = model_like
            case ModelHash() | str():
                model = self._problem.model_hash_to_model(model_like)
            case _:
                raise TypeError(f"Unexpected type: `{type(model_like)}`.")
        return model

    def __setitem__(self, key: ModelIndex, item: ModelLike) -> None:
        match key:
            case int():
                pass
            case ModelHash() | str():
                if key in self._hashes:
                    key = self._hashes.index(key)
                else:
                    key = len(self)
            case slice():
                for key_, item_ in zip(
                    range(*key.indices(len(self))), item, strict=True
                ):
                    self[key_] = item_
            case Iterable():
                for key_, item_ in zip(key, item, strict=True):
                    self[key_] = item_
            case _:
                raise TypeError(f"Unexpected type: `{type(key)}`.")

        item = self._model_like_to_model(model_like=item)

        if key < len(self):
            self._models[key] = item
            self._hashes[key] = item.hash
        else:
            # Key doesn't exist, e.g., instead of
            # models[1] = model1
            # the user did something like
            # models[model1_hash] = model1
            # to add a new model.
            self.append(item)

    def _update(self, index: int, item: ModelLike) -> None:
        """Update the models by adding a new model, with possible replacement.

        If the instance contains a model with a matching hash, that model
        will be replaced.

        Args:
            index:
                The index where the model will be inserted, if it doesn't
                already exist.
            item:
                A model or a model hash.
        """
        model = self._model_like_to_model(item)
        if model.hash in self:
            warnings.warn(
                (
                    f"A model with hash `{model.hash}` already exists "
                    "in this collection of models. The previous model will be "
                    "overwritten."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self[model.hash] = model
        else:
            self._models.insert(index, None)
            self._hashes.insert(index, None)
            # Re-use __setitem__ logic
            self[index] = item

    def __delitem__(self, key: ModelIndex) -> None:
        try:
            match key:
                case ModelHash() | str():
                    key = self._hashes.index(key)
                case slice():
                    for key_ in range(*key.indices(len(self))):
                        del self[key_]
                case Iterable():
                    for key_ in key:
                        del self[key_]
                case _:
                    raise TypeError(f"Unexpected type: `{type(key)}`.")
        except ValueError as err:
            raise KeyError from err

        del self._models[key]
        del self._hashes[key]

    def __add__(
        self, other: ModelLike | ModelsLike, left: bool = True
    ) -> Models:
        match other:
            case Models():
                new_models = other._models
            case Model():
                new_models = [other]
            case ModelHash() | str():
                # Assumes the models belong to the same PEtab Select problem.
                new_models = [self._problem.model_hash_to_model(other)]
            case Iterable():
                # Assumes the models belong to the same PEtab Select problem.
                new_models = Models(
                    models=other, _problem=self._problem
                )._models
            case _:
                raise TypeError(f"Unexpected type: `{type(other)}`.")

        models = self._models + new_models
        if not left:
            models = new_models + self._models
        return Models(models=models, _problem=self._problem)

    def __radd__(self, other: ModelLike | ModelsLike) -> Models:
        return self.__add__(other=other, left=False)

    def __iadd__(self, other: ModelLike | ModelsLike) -> Models:
        return self.__add__(other=other)

    # skipped __mul__, __rmul__, __imul__

    def __copy__(self) -> Models:
        return Models(models=self._models, _problem=self._problem)

    def append(self, item: ModelLike) -> None:
        self._update(index=len(self), item=item)

    def insert(self, index: int, item: ModelLike):
        self._update(index=len(self), item=item)

    # def pop(self, index: int = -1):
    #     model = self._models[index]

    #     # Re-use __delitem__ logic
    #     del self[index]

    #     return model

    # def remove(self, item: ModelLike):
    #     # Re-use __delitem__ logic
    #     if isinstance(item, Model):
    #         item = item.hash
    #     del self[item]

    # skipped clear, copy, count

    def index(self, item: ModelLike, *args) -> int:
        if isinstance(item, Model):
            item = item.hash
        return self._hashes.index(item, *args)

    # skipped reverse, sort

    def extend(self, other: Iterable[ModelLike]) -> None:
        # Re-use append and therein __setitem__ logic
        for model_like in other:
            self.append(model_like)

    def __iter__(self):
        return iter(self._models)

    def __next__(self):
        raise NotImplementedError

    # `dict` methods.

    def get(
        self,
        key: ModelIndex | Iterable[ModelIndex],
        default: ModelLike | None = None,
    ) -> Model | Models:
        try:
            return self[key]
        except KeyError:
            return default

    def values(self) -> Models:
        """Get the models. DEPRECATED."""
        warnings.warn(
            "`models.values()` is deprecated. Use `models` instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self


class Models(_ListDict):
    """A collection of models."""

    def set_problem(self, problem: Problem) -> None:
        """Set the PEtab Select problem for this set of models."""
        self._problem = problem

    def lint(self):
        """Lint the models, e.g. check all hashes are unique.

        Currently raises an exception when invalid.
        """
        duplicates = [
            model_hash
            for model_hash, count in Counter(self._hashes).items()
            if count > 1
        ]
        if duplicates:
            raise ValueError(
                "Multiple models exist with the same hash. "
                f"Model hashes: `{duplicates}`."
            )

    @staticmethod
    def from_yaml(
        filename: TYPE_PATH,
        model_subspace_petab_problem: petab.Problem = None,
        problem: Problem = None,
    ) -> Models:
        """Load models from a YAML file.

        Args:
            filename:
                Location of the YAML file.
            model_subspace_petab_problem:
                A preloaded copy of the PEtab problem. N.B.:
                all models should share the same PEtab problem if this is
                provided (e.g. all models belong to the same model subspace,
                or all model subspaces have the same
                ``model_subspace_petab_yaml`` in the model space file(s)).
            problem:
                The PEtab Select problem. N.B.: all models should belong to the
                same PEtab Select problem if this is provided.

        Returns:
            The models.
        """
        # Handle single-model files, for backwards compatibility.
        try:
            model = Model.from_yaml(
                filename=filename,
                model_subspace_petab_problem=model_subspace_petab_problem,
            )
            return Models([model])
        except:  # noqa: S110
            pass
        return ModelsStandard.load_data(
            filename=filename,
            _problem=problem,
            model_kwargs={
                ROOT_PATH: Path(filename).parent,
                MODEL_SUBSPACE_PETAB_PROBLEM: model_subspace_petab_problem,
            },
        )

    def to_yaml(
        self,
        filename: TYPE_PATH,
        relative_paths: bool = True,
    ) -> None:
        """Save models to a YAML file.

        Args:
            filename:
                Location of the YAML file.
            relative_paths:
                Whether to rewrite the paths in each model (e.g. the path to the
                model's PEtab problem) relative to the ``filename`` location.
        """
        _models = self._models
        if relative_paths:
            root_path = Path(filename).parent
            _models = copy.deepcopy(_models)
            for _model in _models:
                _model.set_relative_paths(root_path=root_path)
        ModelsStandard.save_data(data=Models(_models), filename=filename)

    def get_criterion(
        self,
        criterion: Criterion,
        as_dict: bool = False,
        relative: bool = False,
    ) -> list[float] | dict[ModelHash, float]:
        """Get the criterion value for all models.

        Args:
            criterion:
                The criterion.
            as_dict:
                Whether to return a dictionary, with model hashes for keys.
            relative:
                Whether to compute criterion values relative to the
                smallest criterion value.

        Returns:
            The criterion values.
        """
        result = [model.get_criterion(criterion=criterion) for model in self]
        if relative:
            result = list(np.array(result) - min(result))
        if as_dict:
            result = dict(zip(self._hashes, result, strict=False))
        return result

    def _getattr(
        self,
        attr: str,
        key: Any = None,
        use_default: bool = False,
        default: Any = None,
    ) -> list[Any]:
        """Get an attribute of each model.

        Args:
            attr:
                The name of the attribute (e.g. ``MODEL_ID``).
            key:
                The key of the attribute, if you want to further subset.
                For example, if ``attr=ESTIMATED_PARAMETERS``, this could
                be a specific parameter ID.
            use_default:
                Whether to use a default value for models that are missing
                ``attr`` or ``key``.
            default:
                Value to use for models that do not have ``attr`` or ``key``,
                if ``use_default==True``.

        Returns:
            The list of attribute values.
        """
        # FIXME remove when model is `dataclass`
        values = []
        for model in self:
            try:
                value = getattr(model, attr)
            except:
                if not use_default:
                    raise
                value = default

            if key is not None:
                try:
                    value = value[key]
                except:
                    if not use_default:
                        raise
                    value = default

            values.append(value)
        return values

    @property
    def df(self) -> pd.DataFrame:
        """Get a dataframe of model attributes."""
        return pd.DataFrame(
            {
                MODEL_ID: self._getattr(MODEL_ID),
                MODEL_HASH: self._getattr(MODEL_HASH),
                Criterion.NLLH: self._getattr(
                    CRITERIA, Criterion.NLLH, use_default=True
                ),
                Criterion.AIC: self._getattr(
                    CRITERIA, Criterion.AIC, use_default=True
                ),
                Criterion.AICC: self._getattr(
                    CRITERIA, Criterion.AICC, use_default=True
                ),
                Criterion.BIC: self._getattr(
                    CRITERIA, Criterion.BIC, use_default=True
                ),
                ITERATION: self._getattr(ITERATION, use_default=True),
                PREDECESSOR_MODEL_HASH: self._getattr(
                    PREDECESSOR_MODEL_HASH, use_default=True
                ),
                ESTIMATED_PARAMETERS: self._getattr(
                    ESTIMATED_PARAMETERS, use_default=True
                ),
            }
        )

    @property
    def hashes(self) -> list[ModelHash]:
        return self._hashes


def models_from_yaml_list(
    model_list_yaml: TYPE_PATH,
    petab_problem: petab.Problem = None,
    allow_single_model: bool = True,
    problem: Problem = None,
) -> Models:
    """Deprecated. Use `petab_select.Models.from_yaml` instead."""
    warnings.warn(
        (
            "Use `petab_select.Models.from_yaml` instead. "
            "The `allow_single_model` argument is fixed to `True` now."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return Models.from_yaml(
        filename=model_list_yaml,
        petab_problem=petab_problem,
        problem=problem,
    )


def models_to_yaml_list(
    models: Models,
    output_yaml: TYPE_PATH,
    relative_paths: bool = True,
) -> None:
    """Deprecated. Use `petab_select.Models.to_yaml` instead."""
    warnings.warn(
        "Use `petab_select.Models.to_yaml` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    Models(models=models).to_yaml(
        filename=output_yaml, relative_paths=relative_paths
    )


ModelsStandard = mkstd.YamlStandard(model=Models)
