"""The `ModelSpace` class and related methods."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Iterable
from pathlib import Path
from typing import Any

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
    def load(
        data: TYPE_PATH | pd.DataFrame | list[TYPE_PATH | pd.DataFrame],
        root_path: TYPE_PATH = None,
    ) -> ModelSpace:
        """Load a model space from dataframe(s) or file(s).

        Args:
            data:
                The data. TSV file(s) or pandas dataframe(s).
            root_path:
                Any paths in dataframe will be resolved relative to this path.
                Paths in TSV files will be resolved relative to the directory
                of the TSV file.

        Returns:
            The model space.
        """
        if not isinstance(data, list):
            data = [data]
        dfs = [
            (
                root_path,
                df.reset_index() if df.index.name == MODEL_SUBSPACE_ID else df,
            )
            if isinstance(df, pd.DataFrame)
            else (Path(df).parent, pd.read_csv(df, sep="\t"))
            for df in data
        ]

        model_subspaces = []
        for root_path, df in dfs:
            for _, definition in df.iterrows():
                model_subspaces.append(
                    ModelSubspace.from_definition(
                        definition=definition,
                        root_path=root_path,
                    )
                )
        model_space = ModelSpace(model_subspaces=model_subspaces)
        return model_space

    def save(self, filename: TYPE_PATH | None = None) -> pd.DataFrame:
        """Export the model space to a dataframe (and TSV).

        Args:
            filename:
                If provided, the dataframe will be saved here as a TSV.
                Paths will be made relative to the parent directory of this
                filename.

        Returns:
            The dataframe.
        """
        root_path = Path(filename).parent if filename else None

        data = []
        for model_subspace in self.model_subspaces.values():
            data.append(model_subspace.to_definition(root_path=root_path))
        df = pd.DataFrame(data)
        df = df.set_index(MODEL_SUBSPACE_ID)

        if filename:
            df.to_csv(filename, sep="\t")

        return df

    def search(
        self,
        candidate_space: CandidateSpace,
        limit: int = np.inf,
        exclude: bool = True,
    ):
        """Search all model subspaces according to a candidate space method.

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
