import hashlib

# import json
from typing import Any

from .constants import (  # TYPE_PARAMETER_OPTIONS_DICT,
    ESTIMATE,
    TYPE_PARAMETER_DICT,
    TYPE_PARAMETER_OPTIONS,
)

__all__ = [
    "parameter_string_to_value",
]


def hashify(x: Any, **kwargs) -> str:
    """Generate a hash.

    Currently uses the :func:`hashlib.blake2b` method. `**kwargs` are forwarded
    to this method.

    Args:
        x:
            The object to hash. `x` will be converted to a string (`str(x)`), then
            hashed.

    Returns:
        The hash, as a hexadecimal string.
    """
    # return int(hashlib.sha256(str(x).encode('utf-8')).hexdigest(), 16)
    return hashlib.blake2b(
        str(x).encode("utf-8"),
        **kwargs,
    ).hexdigest()


def hash_parameter_dict(dict_: TYPE_PARAMETER_DICT, **kwargs):
    """Hash a dictionary of parameter values."""
    value = tuple((k, dict_[k]) for k in sorted(dict_))
    return hashify(value, **kwargs)


def hash_parameter_options(list_: TYPE_PARAMETER_OPTIONS, **kwargs):
    """Hash parameter options."""
    return hashify(list(list_), **kwargs)


def hash_str(str_: str, **kwargs):
    return hashify(str_, **kwargs)


def hash_list(list_: list, **kwargs):
    return hashify(list(list_), **kwargs)


def snake_case_to_camel_case(string: str) -> str:
    """Convert a string from snake case to camel case.

    Args:
        string:
            The string, in snake case.

    Returns:
        The string, in camel case.
    """
    string_pieces = string.split("_")
    string_camel = ""
    for string_piece in string_pieces:
        string_camel += string_piece[0].upper() + string_piece[1:]
    return string_camel


def parameter_string_to_value(
    parameter_string: str,
    passthrough_estimate: bool = False,
) -> float | int | str:
    """Cast a parameter value from string to numeric.

    Args:
        parameter_string:
            The parameter value, as a string.
        passthrough_estimate:
            Whether to return `ESTIMATE` as `ESTIMATE`. If `False`, raises an exception
            if `parameter_string == ESTIMATE`.

    Returns:
        The parameter value, as a numeric type.
    """
    if parameter_string == ESTIMATE:
        if passthrough_estimate:
            return parameter_string
        raise ValueError("Please handle estimated parameters differently.")
    float_value = float(parameter_string)
    int_value = int(float_value)

    if int_value == float_value:
        return int_value
    return float_value


# def hash_dictionary(dictionary: Dict[str, Union[]]):
#    return hash(json.dumps(dictionary, sort_keys=True))
#
#
# def hash_list(list_: List):
#    return hash(json.dumps(list_, sort_keys=True))
