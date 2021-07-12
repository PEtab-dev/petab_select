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
ESTIMATE_SYMBOL_INTERNAL = 'nan'

TYPING_PATH = Union[str, Path]

# Model specification row columns
# TODO ensure none of these occur twice in the column header (this would
#      suggest that a parameter has a conflicting name)
#MODEL_ID = 'modelId'  # TODO already defined, reorganize constants
#YAML = 'YAML'  # FIXME
MODEL_ID = 'model_id'
PETAB_YAML = 'petab_yaml'  # FIXME
SBML = 'sbml'
HASH = 'hash'

MODEL_SPACE_SPECIFICATION_NOT_PARAMETERS = [MODEL_ID, PETAB_YAML]

COMPARED_MODEL_ID = 'compared_'+MODEL_ID
YAML_FILENAME = 'yaml'

#FORWARD = 'forward'
#BACKWARD = 'backward'
#BIDIRECTIONAL = 'bidirectional'
#LATERAL = 'lateral'

BIDIRECTIONAL = 'bidirectional'
FORWARD = 'forward'
BACKWARD = 'backward'
LATERAL = 'lateral'

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
ESTIMATE_PARAMETERS = 'estimated_parameters'

CRITERION = 'criterion'
METHOD = 'method'
VERSION = 'version'
MODEL_SPECIFICATION_FILES = 'model_specification_files'
