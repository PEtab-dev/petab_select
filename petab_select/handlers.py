from pathlib import Path
from typing import Any, Callable, List, Optional, Union

import numpy as np

# `float` for `np.inf`
TYPE_LIMIT = Union[float, int]


# TODO exclusions handler


class LimitHandler():
    """A handler for classes that have a limit.

    Attributes:
        current:
            A callable to determine the current value.
        limit:
            The upper limit of the current value.
    """

    def __init__(
        self,
        current: Callable[[], bool],
        limit: TYPE_LIMIT,
    ):
        self.current = current
        self.limit = limit

    def reached(self) -> bool:
        """Check whether the limit has been reached."""
        if self.current() >= self.limit:
            return True
        return False

    def set_limit(self, limit: TYPE_LIMIT) -> None:
        """Set the limit.

        Args:
            limit:
                The limit.
        """
        self.limit = limit

    def get_limit(self) -> TYPE_LIMIT:
        """Get the limit.

        Returns:
            The limit.
        """
        return self.limit

