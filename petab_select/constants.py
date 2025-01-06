"""Constants for the PEtab Select package."""

from __future__ import annotations

import string
import sys
from enum import Enum
from pathlib import Path
from typing import Literal

# Checked

# Criteria
CRITERIA = "criteria"
CRITERION = "criterion"


class Criterion(str, Enum):
    """String literals for model selection criteria."""

    #: The Akaike information criterion.
    AIC = "AIC"
    #: The corrected Akaike information criterion.
    AICC = "AICc"
    #: The Bayesian information criterion.
    BIC = "BIC"
    #: The likelihood.
    LH = "LH"
    #: The log-likelihood.
    LLH = "LLH"
    #: The negative log-likelihood.
    NLLH = "NLLH"
    #: The sum of squared residuals.
    SSR = "SSR"


# Model
ESTIMATED_PARAMETERS = "estimated_parameters"
ITERATION = "iteration"
MODEL_ID = "model_id"
MODEL_SUBSPACE_ID = "model_subspace_id"
MODEL_SUBSPACE_INDICES = "model_subspace_indices"
PARAMETERS = "parameters"
MODEL_SUBSPACE_PETAB_YAML = "model_subspace_petab_yaml"
MODEL_SUBSPACE_PETAB_PROBLEM = "_model_subspace_petab_problem"
PETAB_YAML = "petab_yaml"
ROOT_PATH = "root_path"
ESTIMATE = "estimate"

PETAB_PROBLEM = "petab_problem"

# Model hash
MODEL_HASH = "model_hash"
MODEL_HASH_DELIMITER = "-"
MODEL_SUBSPACE_INDICES_HASH = "model_subspace_indices_hash"
MODEL_SUBSPACE_INDICES_HASH_DELIMITER = "."
MODEL_SUBSPACE_INDICES_HASH_MAP = (
    # [0-9]+[A-Z]+[a-z]
    string.digits + string.ascii_uppercase + string.ascii_lowercase
)
PREDECESSOR_MODEL_HASH = "predecessor_model_hash"
ITERATION = "iteration"
PETAB_PROBLEM = "petab_problem"
PETAB_YAML = "petab_yaml"
HASH = "hash"

# MODEL_SPACE_FILE_NON_PARAMETER_COLUMNS = [MODEL_ID, PETAB_YAML]
MODEL_SPACE_FILE_NON_PARAMETER_COLUMNS = [MODEL_SUBSPACE_ID, PETAB_YAML]

# PEtab
PETAB_ESTIMATE_TRUE = 1

# Problem
MODEL_SPACE_FILES = "model_space_files"
PROBLEM = "problem"
VERSION = "version"

# Candidate space
CANDIDATE_SPACE = "candidate_space"
CANDIDATE_SPACE_ARGUMENTS = "candidate_space_arguments"
METHOD = "method"
METHOD_SCHEME = "method_scheme"
NEXT_METHOD = "next_method"
PREVIOUS_METHODS = "previous_methods"
PREDECESSOR_MODEL = "predecessor_model"


class Method(str, Enum):
    """String literals for model selection methods."""

    #: The backward stepwise method.
    BACKWARD = "backward"
    #: The brute-force method.
    BRUTE_FORCE = "brute_force"
    #: The FAMoS method.
    FAMOS = "famos"
    #: The forward stepwise method.
    FORWARD = "forward"
    #: The lateral, or swap, method.
    LATERAL = "lateral"
    #: The jump-to-most-distant-model method.
    MOST_DISTANT = "most_distant"


# Typing
TYPE_PATH = str | Path

# UI
MODELS = "models"
UNCALIBRATED_MODELS = "uncalibrated_models"
TERMINATE = "terminate"

#: Methods that move through model space by taking steps away from some model.
STEPWISE_METHODS = [
    Method.BACKWARD,
    Method.FORWARD,
    Method.LATERAL,
]
#: Methods that require an initial model.
INITIAL_MODEL_METHODS = [
    Method.BACKWARD,
    Method.FORWARD,
    Method.LATERAL,
]

#: Virtual initial models can be used to initialize some initial model methods.
# FIXME replace by real "dummy" model object
# VIRTUAL_INITIAL_MODEL = "virtual_initial_model"
#: Methods that are compatible with a virtual initial model.
VIRTUAL_INITIAL_MODEL_METHODS = [
    Method.BACKWARD,
    Method.FORWARD,
]


__all__ = [
    x
    for x in dir(sys.modules[__name__])
    if not x.startswith("_")
    and x not in ("sys", "Enum", "Path", "Dict", "List", "Literal", "Union")
]


# Unchecked
MODEL = "model"

# Zero-indexed column/row indices
MODEL_ID_COLUMN = 0
PETAB_YAML_COLUMN = 1
# It is assumed that all columns after PARAMETER_DEFINITIONS_START contain
# parameter IDs.
PARAMETER_DEFINITIONS_START = 2
HEADER_ROW = 0

PARAMETER_VALUE_DELIMITER = ";"
CODE_DELIMITER = "-"
PETAB_ESTIMATE_FALSE = 0

# TYPING_PATH = Union[str, Path]

# Model space file columns
# TODO ensure none of these occur twice in the column header (this would
#      suggest that a parameter has a conflicting name)
# MODEL_ID = 'modelId'  # TODO already defined, reorganize constants
# YAML = 'YAML'  # FIXME
MODEL_CODE = "model_code"
MODEL_HASHES = "model_hashes"
PETAB_HASH_DIGEST_SIZE = None
# If `predecessor_model_hash` is defined for a model, it is the ID of the model that the
# current model was/is to be compared to. This is part of the result and is
# only (optionally) set by the PEtab calibration tool. It is not defined by the
# PEtab Select model selection problem (but may be subsequently stored in the
# PEtab Select model report format.
HASH = "hash"

YAML_FILENAME = "yaml"

# DISTANCES = {
#    FORWARD: {
#        'l1': 1,
#        'size': 1,
#    },
#    BACKWARD: {
#        'l1': 1,
#        'size': -1,
#    },
#    LATERAL: {
#        'l1': 2,
#        'size': 0,
#    },
# }


# Parameters can be fixed to a value, or estimated if indicated with the string
# `ESTIMATE`.
TYPE_PARAMETER = float | int | Literal[ESTIMATE]
TYPE_PARAMETER_OPTIONS = list[TYPE_PARAMETER]
# Parameter ID -> parameter value mapping.
TYPE_PARAMETER_DICT = dict[str, TYPE_PARAMETER]
# Parameter ID -> multiple possible parameter values.
TYPE_PARAMETER_OPTIONS_DICT = dict[str, TYPE_PARAMETER_OPTIONS]

TYPE_CRITERION = float
