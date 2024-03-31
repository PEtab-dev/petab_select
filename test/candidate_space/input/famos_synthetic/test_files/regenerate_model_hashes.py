import pandas as pd

import petab_select

petab_select_problem = petab_select.Problem.from_yaml(
    "../select/FAMoS_2019_petab_select_problem.yaml"
)
df = pd.read_csv("calibration_results.tsv", sep="\t", dtype=str)

model_hashes = list(df.model_hash)
new_model_hashes = [
    petab_select_problem.get_model(
        model_subspace_id="model_subspace_1",
        model_subspace_indices=[
            int(s)
            for s in petab_select.model.ModelHash.from_hash(
                model_hash
            ).unhash_model_subspace_indices()
        ],
    ).get_hash()
    for model_hash in model_hashes
]

df["model_hash"] = new_model_hashes
df.to_csv("D.tsv", sep="\t", index=False)