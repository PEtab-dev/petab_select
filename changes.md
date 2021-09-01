# Changes
- Renamed the `search` method of the CLI to `candidates`
- Added selection ability
  - PEtab Select will initialize a model search method (e.g. the forward method) with the model with the best criterion value, from a list of calibrated models in the report format
  - CLI interface: `-b`/`--best` argument
  - Python interface: `petab_select.Problem.get_best`
- Renamed "model specification files" to "model space files" for clarity
- Renamed `model0_id` to `predecessor_model_id` for readability
- The file format for initial models, model interchange, and reporting is combined into a single "model" format, with some optional keys that are required for different tasks (see README)
- Added `model2petab` and `models2petab` methods to the CLI

# Upcoming changes
- Might rename the `model_id` column in model space files to `model_set_id_prefix` for clarity (as a row can encode multiple models in the compressed format)
  - Might then create `model_id` reproducibly as a combination of `model_set_id_prefix` + a hash of the model
- Implement the `predecessor_model_id`
  - Produce graphs given a set of models
