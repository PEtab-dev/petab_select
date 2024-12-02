from __future__ import annotations

import warnings
from collections import Counter
from collections.abc import Iterable, MutableSequence
from pathlib import Path
from typing import TYPE_CHECKING, TypeAlias

import yaml

from .constants import TYPE_PATH
from .model import (
    Model,
    ModelHash,
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
    "ListDict",
    "Models",
    "models_from_yaml_list",
    "models_to_yaml_list",
]


class ListDict(MutableSequence):
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

    Attributes:
        _models:
            The list of objects (list items/dictionary values)
            (PEtab Select models).
        _hashes:
            The list of metadata (dictionary keys) (model hashes).
        _problem:
    """

    def __init__(
        self, models: Iterable[ModelLike] = None, problem: Problem = None
    ) -> Models:
        self._models = []
        self._hashes = []
        self._problem = problem

        if models is None:
            models = []
        self.extend(models)

    def __repr__(self) -> str:
        """Get the model hashes that can regenerate these models.

        N.B.: some information, e.g. criteria, will be lost if the hashes are
        used to reproduce the set of models.
        """
        return repr(self._hashes)

    # skipped __lt__, __le__

    def __eq__(self, other) -> bool:
        other_hashes = Models(other)._hashes
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
                    print(key)
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
            self._hashes[key] = item.get_hash()
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
        if model.get_hash() in self:
            warnings.warn(
                (
                    f"A model with hash `{model.get_hash()}` already exists "
                    "in this collection of models. The previous model will be "
                    "overwritten."
                ),
                RuntimeWarning,
                stacklevel=2,
            )
            self[model.get_hash()] = model
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
                new_models = Models(other, problem=self._problem)._models
            case _:
                raise TypeError(f"Unexpected type: `{type(other)}`.")

        models = self._models + new_models
        if not left:
            models = new_models + self._models
        return Models(models=models, problem=self._problem)

    def __radd__(self, other: ModelLike | ModelsLike) -> Models:
        return self.__add__(other=other, left=False)

    def __iadd__(self, other: ModelLike | ModelsLike) -> Models:
        return self.__add__(other=other)

    # skipped __mul__, __rmul__, __imul__

    def __copy__(self) -> Models:
        return Models(models=self._models, problem=self._problem)

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
    #         item = item.get_hash()
    #     del self[item]

    # skipped clear, copy, count

    def index(self, item: ModelLike, *args) -> int:
        if isinstance(item, Model):
            item = item.get_hash()
        return self._hashes.index(item, *args)

    # skipped reverse, sort

    def extend(self, other: Iterable[ModelLike]) -> None:
        # Re-use append and therein __setitem__ logic
        for model_like in other:
            self.append(model_like)

    # __iter__/__next__? Not in UserList...

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


class Models(ListDict):
    """A collection of models.

    Provide a PEtab Select ``problem`` to the constructor or via
    ``set_problem``, to use add models by hashes. This means that all models
    must belong to the same PEtab Select problem.

    This permits both ``list`` and ``dict`` operations -- see
    :class:``ListDict`` for further details.
    """

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
        models_yaml: TYPE_PATH,
        petab_problem: petab.Problem = None,
        problem: Problem = None,
    ) -> Models:
        """Generate models from a PEtab Select list of model YAML file.

        Args:
            models_yaml:
                The path to the PEtab Select list of model YAML file.
            petab_problem:
                See :meth:`Model.from_dict`.
            problem:
                The PEtab Select problem.

        Returns:
            The models.
        """
        with open(str(models_yaml)) as f:
            model_dict_list = yaml.safe_load(f)
        if not model_dict_list:
            # Empty file
            models = []
        elif not isinstance(model_dict_list, list):
            # File contains a single model
            models = [
                Model.from_dict(
                    model_dict_list,
                    base_path=Path(models_yaml).parent,
                    petab_problem=petab_problem,
                )
            ]
        else:
            # File contains a list of models
            models = [
                Model.from_dict(
                    model_dict,
                    base_path=Path(models_yaml).parent,
                    petab_problem=petab_problem,
                )
                for model_dict in model_dict_list
            ]

        return Models(models=models, problem=problem)

    def to_yaml(
        self,
        output_yaml: TYPE_PATH,
        relative_paths: bool = True,
    ) -> None:
        """Generate a YAML listing of models.

        Args:
            output_yaml:
                The location where the YAML will be saved.
            relative_paths:
                Whether to rewrite the paths in each model (e.g. the path to the
                model's PEtab problem) relative to the `output_yaml` location.
        """
        paths_relative_to = None
        if relative_paths:
            paths_relative_to = Path(output_yaml).parent
        model_dicts = [
            model.to_dict(paths_relative_to=paths_relative_to)
            for model in self
        ]
        with open(output_yaml, "w") as f:
            yaml.safe_dump(model_dicts, f)


def models_from_yaml_list(
    model_list_yaml: TYPE_PATH,
    petab_problem: petab.Problem = None,
    allow_single_model: bool = True,
    problem: Problem = None,
) -> Models:
    """Generate a model from a PEtab Select list of model YAML file.

    Deprecated. Use `petab_select.Models.from_yaml` instead.

    Args:
        model_list_yaml:
            The path to the PEtab Select list of model YAML file.
        petab_problem:
            See :meth:`Model.from_dict`.
        allow_single_model:
            Given a YAML file that contains a single model directly (not in
            a 1-element list), if ``True`` then the single model will be read in,
            else a ``ValueError`` will be raised.
        problem:
            The PEtab Select problem.

    Returns:
        The models.
    """
    warnings.warn(
        (
            "Use `petab_select.Models.from_yaml` instead. "
            "The `allow_single_model` argument is fixed to `True` now."
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return Models.from_yaml(
        models_yaml=model_list_yaml,
        petab_problem=petab_problem,
        problem=problem,
    )


def models_to_yaml_list(
    models: Models,
    output_yaml: TYPE_PATH,
    relative_paths: bool = True,
) -> None:
    """Generate a YAML listing of models.

    Deprecated. Use `petab_select.Models.to_yaml` instead.

    Args:
        models:
            The models.
        output_yaml:
            The location where the YAML will be saved.
        relative_paths:
            Whether to rewrite the paths in each model (e.g. the path to the
            model's PEtab problem) relative to the `output_yaml` location.
    """
    warnings.warn(
        "Use `petab_select.Models.to_yaml` instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    Models(models=models).to_yaml(
        output_yaml=output_yaml, relative_paths=relative_paths
    )
