"""Constants for the PEtab Select package."""
from enum import Enum
from pathlib import Path
from typing import Dict, List, Union


# Zero-indexed column/row indices
MODEL_ID_COLUMN = 0
PETAB_YAML_COLUMN = 1
# It is assumed that all columns after PARAMETER_DEFINITIONS_START contain
# parameter IDs.
PARAMETER_DEFINITIONS_START = 2
HEADER_ROW = 0

PARAMETER_VALUE_DELIMITER = ';'
CODE_DELIMITER = '-'
ESTIMATE = 'estimate'
PETAB_ESTIMATE_FALSE = 0
PETAB_ESTIMATE_TRUE = 1

#TYPING_PATH = Union[str, Path]
TYPE_PATH = Union[str, Path]

# Model space file columns
# TODO ensure none of these occur twice in the column header (this would
#      suggest that a parameter has a conflicting name)
#MODEL_ID = 'modelId'  # TODO already defined, reorganize constants
#YAML = 'YAML'  # FIXME
MODEL_ID = 'model_id'
MODEL_SUBSPACE_ID = 'model_subspace_id'
MODEL_SUBSPACE_INDICES = 'model_subspace_indices'
MODEL_CODE = 'model_code'
MODEL_HASH = 'model_hash'
# If `predecessor_model_id` is defined for a model, it is the ID of the model that the
# current model was/is to be compared to. This is part of the result and is
# only (optionally) set by the PEtab calibration tool. It is not defined by the
# PEtab Select model selection problem (but may be subsequently stored in the
# PEtab Select model report format.
PREDECESSOR_MODEL_ID = 'predecessor_model_id'
PETAB_PROBLEM = 'petab_problem'
PETAB_YAML = 'petab_yaml'
SBML = 'sbml'
HASH = 'hash'

#MODEL_SPACE_FILE_NON_PARAMETER_COLUMNS = [MODEL_ID, PETAB_YAML]
MODEL_SPACE_FILE_NON_PARAMETER_COLUMNS = [MODEL_SUBSPACE_ID, PETAB_YAML]

#COMPARED_MODEL_ID = 'compared_'+MODEL_ID
YAML_FILENAME = 'yaml'

#FORWARD = 'forward'
#BACKWARD = 'backward'
#BIDIRECTIONAL = 'bidirectional'
#LATERAL = 'lateral'

BIDIRECTIONAL = 'bidirectional'
FORWARD = 'forward'
BACKWARD = 'backward'
LATERAL = 'lateral'
BRUTE_FORCE = 'brute_force'
# Methods that move through model space by taking steps away from some model.
STEPWISE_METHODS = [
    BACKWARD,
    BIDIRECTIONAL,
    FORWARD,
    LATERAL,
]
# Methods that require an initial model.
INITIAL_MODEL_METHODS = [
    BACKWARD,
    FORWARD,
    LATERAL,
]

# Virtual initial models can be used to initialize some initial model methods.
VIRTUAL_INITIAL_MODEL = 'virtual_initial_model'
VIRTUAL_INITIAL_MODEL_METHODS = [
    FORWARD,
    BACKWARD
]

#DISTANCES = {
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
#}

CRITERIA = 'criteria'
# FIXME remove, change all uses to Enum below
#AIC = 'AIC'
#AICC = 'AICc'
#BIC = 'BIC'
#AKAIKE_INFORMATION_CRITERION = AIC
#CORRECTED_AKAIKE_INFORMATION_CRITERION = AICC
#BAYESIAN_INFORMATION_CRITERION = BIC
#LH = 'LH'
#LLH = 'LLH'
#NLLH = 'NLLH'
#LIKELIHOOD = LH
#LOG_LIKELIHOOD = LLH
#NEGATIVE_LOG_LIKELIHOOD = NLLH


PARAMETERS = 'parameters'
#PARAMETER_ESTIMATE = 'parameter_estimate'
ESTIMATED_PARAMETERS = 'estimated_parameters'

CRITERION = 'criterion'
METHOD = 'method'
VERSION = 'version'
MODEL_SPACE_FILES = 'model_space_files'

MODEL = 'model'

# Parameters can be fixed to a value, or estimated if indicated with the string
# `ESTIMATE`.
# TODO change to `Literal[ESTIMATE]` (Python >= 3.8)
TYPE_PARAMETER = Union[float, int, ESTIMATE]
TYPE_PARAMETER_OPTIONS = List[TYPE_PARAMETER]
# Parameter ID -> parameter value mapping.
TYPE_PARAMETER_DICT = Dict[str, TYPE_PARAMETER]
# Parameter ID -> multiple possible parameter values.
TYPE_PARAMETER_OPTIONS_DICT = Dict[str, TYPE_PARAMETER_OPTIONS]

TYPE_CRITERION = float


class Method(str, Enum):
    """String literals for model selection methods."""
    BACKWARD = 'backward'
    BIDIRECTIONAL = 'bidirectional'
    BRUTE_FORCE = 'brute_force'
    FORWARD = 'forward'
    LATERAL = 'lateral'


class Criterion(str, Enum):
    """String literals for model selection criteria."""
    AIC = 'AIC'
    AICC = 'AICc'
    BIC = 'BIC'
    LH = 'LH'
    LLH = 'LLH'
    NLLH = 'NLLH'
