import sys

from example_cli_famos_helpers import calibrate

import petab_select

models_yaml = sys.argv[1]
calibrated_models_yaml = sys.argv[2]

models = petab_select.model.models_from_yaml_list(models_yaml)
predecessor_model_hashes = set()
for model in models:
    calibrate(model=model)
    predecessor_model_hashes |= {model.predecessor_model_hash}
petab_select.model.models_to_yaml_list(
    models=models, output_yaml=calibrated_models_yaml
)

if len(predecessor_model_hashes) == 0:
    pass
elif len(predecessor_model_hashes) == 1:
    (predecessor_model_hash,) = predecessor_model_hashes
else:
    print(
        "The models of this iteration somehow have different predecessor models.\n"
        + "\n".join(predecessor_model_hashes)
    )
