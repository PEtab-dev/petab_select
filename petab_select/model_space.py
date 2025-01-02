"""The `ModelSpace` class and related methods."""

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any, get_args

import numpy as np
import pandas as pd

from .candidate_space import CandidateSpace
from .constants import (
    MODEL_SUBSPACE_ID,
    TYPE_PATH,
)
from .model import Model
from .model_subspace import ModelSubspace

__all__ = [
    "ModelSpace",
    "get_model_space_df",
]


class ModelSpace:
    """A model space, as a collection of model subspaces.

    Attributes:
        model_subspaces:
            List of model subspaces.
        exclusions:
            Hashes of models that are excluded from the model space.
    """

    def __init__(
        self,
        model_subspaces: list[ModelSubspace],
    ):
        self.model_subspaces = {
            model_subspace.model_subspace_id: model_subspace
            for model_subspace in model_subspaces
        }

    @staticmethod
    def from_files(
        filenames: list[TYPE_PATH],
    ):
        """Create a model space from model space files.

        Args:
            filenames:
                The locations of the model space files.

        Returns:
            The corresponding model space.
        """
        # TODO validate input?
        model_space_dfs = [
            get_model_space_df(filename) for filename in filenames
        ]
        model_subspaces = []
        for model_space_df, model_space_filename in zip(
            model_space_dfs, filenames, strict=False
        ):
            for model_subspace_id, definition in model_space_df.iterrows():
                model_subspaces.append(
                    ModelSubspace.from_definition(
                        model_subspace_id=model_subspace_id,
                        definition=definition,
                        parent_path=Path(model_space_filename).parent,
                    )
                )
        model_space = ModelSpace(model_subspaces=model_subspaces)
        return model_space

    @staticmethod
    def from_df(
        df: pd.DataFrame,
        parent_path: TYPE_PATH = None,
    ):
        model_subspaces = []
        for model_subspace_id, definition in df.iterrows():
            model_subspaces.append(
                ModelSubspace.from_definition(
                    model_subspace_id=model_subspace_id,
                    definition=definition,
                    parent_path=parent_path,
                )
            )
        model_space = ModelSpace(model_subspaces=model_subspaces)
        return model_space

    # TODO: `to_df` / `to_file`

    def search(
        self,
        candidate_space: CandidateSpace,
        limit: int = np.inf,
        exclude: bool = True,
    ):
        """...TODO

        Args:
            candidate_space:
                The candidate space.
            limit:
                The maximum number of models to send to the candidate space (i.e. this
                limit is on the number of models considered, not necessarily approved
                as candidates).
                Note that using a limit may produce unexpected results. For
                example, it may bias candidate models to be chosen only from
                a subset of model subspaces.
            exclude:
                Whether to exclude the new candidates from the model subspaces.
        """
        if candidate_space.limit.reached():
            warnings.warn(
                "The candidate space has already reached its limit of accepted models.",
                RuntimeWarning,
                stacklevel=2,
            )
            return candidate_space.models

        @candidate_space.wrap_search_subspaces
        def search_subspaces(only_one_subspace: bool = False):
            # TODO change dict to list of subspaces. Each subspace should manage its own
            #      ID
            if only_one_subspace and len(self.model_subspaces) > 1:
                logging.warning(
                    f"There is more than one model subspace. This can lead to problems for candidate space {candidate_space}, especially if they have different PEtab YAML files."
                )
            for model_subspace in self.model_subspaces.values():
                model_subspace.search(
                    candidate_space=candidate_space, limit=limit
                )
                if len(candidate_space.models) == limit:
                    break
                elif len(candidate_space.models) > limit:
                    raise ValueError(
                        "An unknown error has occurred. Too many models were "
                        f"generated. Requested limit: {limit}. Number of "
                        f"generated models: {len(candidate_space.models)}."
                    )

        search_subspaces()

        ## FIXME implement source_path.. somewhere
        # if self.source_path is not None:
        #    for model in candidate_space.models:
        #        # TODO do this change elsewhere instead?
        #        # e.g. model subspace
        #        model.petab_yaml = self.source_path / model.petab_yaml

        if exclude:
            self.exclude_models(candidate_space.models)

        return candidate_space.models

    def __len__(self):
        """Get the number of models in this space."""
        subspace_counts = [len(s) for s in self.model_subspaces]
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
            # model_subspace.reset_exclusions()

    def exclude_model_hashes(self, model_hashes: Iterable[str]):
        # FIXME add Exclusions Mixin (or object) to handle exclusions on the subspace
        # and space level.
        for model_subspace in self.model_subspaces.values():
            model_subspace.exclude_model_hashes(model_hashes=model_hashes)

    def reset_exclusions(
        self,
        exclusions: list[Any] | None | None = None,
    ) -> None:
        """Reset the exclusions in the model subspaces."""
        for model_subspace in self.model_subspaces.values():
            model_subspace.reset_exclusions(exclusions)


def get_model_space_df(df: TYPE_PATH | pd.DataFrame) -> pd.DataFrame:
    """Get a model space dataframe.

    Args:
        The model space.

    Returns:
        The model space, appropriately formatted.
    """
    if isinstance(df, get_args(TYPE_PATH)):
        df = pd.read_csv(df, sep="\t")
    if df.index.name != MODEL_SUBSPACE_ID:
        df.set_index([MODEL_SUBSPACE_ID], inplace=True)
    return df
