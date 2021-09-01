"""Constants for the PEtab Select package."""
from pathlib import Path
from typing import Union


# Zero-indexed column/row indices
MODEL_ID_COLUMN = 0
PETAB_YAML_COLUMN = 1
# It is assumed that all columns after PARAMETER_DEFINITIONS_START contain
# parameter IDs.
PARAMETER_DEFINITIONS_START = 2
HEADER_ROW = 0

PARAMETER_VALUE_DELIMITER = ';'
ESTIMATE = 'estimate'
ESTIMATE_SYMBOL_UI = ESTIMATE
# Here, 'nan' is a string as it will be written to a (temporary) file. The
# actual internal symbol is float('nan'). Equality to this symbol should be
# checked with a function like `math.isnan()` (not ` == float('nan')`).
ESTIMATE_SYMBOL_INTERNAL_STR = 'nan'
ESTIMATE_SYMBOL_INTERNAL = float(ESTIMATE_SYMBOL_INTERNAL_STR)

TYPING_PATH = Union[str, Path]

# Model space file columns
# TODO ensure none of these occur twice in the column header (this would
#      suggest that a parameter has a conflicting name)
#MODEL_ID = 'modelId'  # TODO already defined, reorganize constants
#YAML = 'YAML'  # FIXME
MODEL_ID = 'model_id'
# If `model0_id` is defined for a model, it is the ID of the model that the
# current model was/is to be compared to. This is part of the result and is
# only (optionally) set by the PEtab calibration tool. It is not defined by the
# PEtab Select model selection problem (but may be subsequently stored in the
# PEtab Select model report format.
PREDECESSOR_MODEL_ID = 'predecessor_model_id'
PETAB_YAML = 'petab_yaml'  # FIXME
SBML = 'sbml'
HASH = 'hash'

MODEL_SPACE_FILE_NON_PARAMETER_COLUMNS = [MODEL_ID, PETAB_YAML]

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
# Methods that require an initial model.
INITIAL_MODEL_METHODS = [
    BACKWARD,
    FORWARD,
    LATERAL,
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
AIC = 'AIC'
AICC = 'AICc'
BIC = 'BIC'

PARAMETERS = 'parameters'
#PARAMETER_ESTIMATE = 'parameter_estimate'
ESTIMATED_PARAMETERS = 'estimated_parameters'

CRITERION = 'criterion'
METHOD = 'method'
VERSION = 'version'
MODEL_SPACE_FILES = 'model_space_files'

MODEL = 'model'
