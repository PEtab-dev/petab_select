import hashlib
#import json
from typing import Any, Dict, List, Union

from .constants import (
    ESTIMATE,
    TYPE_PARAMETER_DICT,
    TYPE_PARAMETER_OPTIONS,
    #TYPE_PARAMETER_OPTIONS_DICT,
)


def hashify(x: Any) -> str:
    """Generate a hash.

    Args:
        x:
            The object to hash. `x` will be converted to a string (`str(x)`), then
            hashed.

    Returns:
        The hash.
    """
    #return int(hashlib.sha256(str(x).encode('utf-8')).hexdigest(), 16)
    return hashlib.blake2b(str(x).encode('utf-8')).hexdigest()


def hash_parameter_dict(dict_: TYPE_PARAMETER_DICT):
    """Hash a dictionary of parameter values."""
    value = tuple(zip(dict_.keys(), dict_.values()))
    return hashify(value)


def hash_parameter_options(list_: TYPE_PARAMETER_OPTIONS):
    """Hash parameter options."""
    return hashify(list(list_))


def hash_str(str_: str):
    return hashify(str_)


def hash_list(list_: List):
    return hashify(list(list_))


def snake_case_to_camel_case(string: str) -> str:
    """Convert a string from snake case to camel case.

    Args:
        string:
            The string, in snake case.

    Returns:
        The string, in camel case.
    """
    string_pieces = string.split('_')
    string_camel = ''
    for string_piece in string_pieces:
        string_camel += string_piece[0].upper() + string_piece[1:]
    return string_camel


def parameter_string_to_value(
    parameter_string: str,
    passthrough_estimate: bool = False,
) -> Union[float, int, str]:
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
        raise ValueError('Please handle estimated parameters differently.')
    float_value = float(parameter_string)
    int_value = int(float_value)

    if int_value == float_value:
        return int_value
    return float_value



#def hash_dictionary(dictionary: Dict[str, Union[]]):
#    return hash(json.dumps(dictionary, sort_keys=True))
#
#
#def hash_list(list_: List):
#    return hash(json.dumps(list_, sort_keys=True))
