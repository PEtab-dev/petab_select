import json
from typing import Dict


def hash_dictionary(dictionary: Dict):
    return hash(json.dumps(dictionary, sort_keys=True))
