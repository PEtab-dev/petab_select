# Changes

## 0.0.9
Fixed automatic exclusions from the Python interface. Now, exclusions are only automatically managed in `Problem`s (thereby also the model space and model subspaces) and `CandidateSpace`s.

## 0.0.8
Previously, use of a `limit` via the UI (e.g. CLI) limited the number of models sent to the candidate space. The candidate space may then reject some models, which means this limit is not the number of models seen in the output.

This version has a breaking change. The `limit` is now the number of models in the output (if the model space has sufficient acceptable models to reach the limit). This `limit` now matches the description that was already provided in the Python and CLI docs.

The behavior of the previous `limit` is now implemented as `limit_sent` in the Python or CLI UI.

## 0.0.7 (bugfix)
- fix implementation of limit in model subspace (GitHub issue #3)

## 0.0.6 (bugfix)
- fixed calculation of number of priors (GitHub issue #2)

## 0.0.5
- renamed `initial_model` to `predecessor_model` in the UI method `candidates`, to avoid confusion.
  - "Initial model" is used to refer to the first model used in any algorithm. "Predecessor model" is the first model in a specific iteration of the algorithm. For the first iteration, initial model and predecessor model are the same.
- added `excluded_model_hashes` as an option to the `candidates` UI method (an alternative to `excluded_models`)
  - named `excluded_model_hash_files` in the CLI -- files with one model hash per line
  - **breaking** renamed `exclude_models` to `excluded_model_files` for consistency / avoid naming conflicts
- the `LateralCandidateSpace` now explicitly requires a predecessor model
- getters and setters for the predecessor model, model exclusions, and limit on number of models, added to `CandidateSpace`
- renamed files in the test cases for consistency

## 0.0.4 (bugfix)

## 0.0.3
- **Added model subspaces**
  - efficiently searches for candidate models in stepwise methods
  - handles exponentially large model spaces
- **Virtual initial models** are now automatically generated if no initial model is provided, for methods that require an initial model
- Refactored methods in `criteria.py` -- now has a `CriteriaCalculator`
  - **PEtab problem calibrators only need to provide "sufficient" information** (just one of the following) in order to have criteria computed by PEtab Select
    - likelihood
    - log-likelihood
    - negative log-likelihood
  - includes methods to transform likelihood
  - should all for calculation of all possible criteria, given some initial subset of criteria
    - e.g., providing a model and its likelihood allows computation of
      - log-likelihood
      - negative log-likelihood
      - AIC
      - AICc
      - BIC
- Added `petab_select best` to the CLI, to get the best model from a list of calibrated models.
- Replaced the symbol for estimated parameters everywhere
  - was `ESTIMATE_SYMBOL_UI==np.nan` in some cases -- now use `ESTIMATE=='estimate'`
  - the following are removed and replaced with `ESTIMATE` in both the file format and Python package
    - `ESTIMATE_SYMBOL_UI`
    - `ESTIMATE_SYMBOL_INTERNAL`
    - `ESTIMATE_SYMBOL_INTERNAL_STR`
- More tests
- CLI (methods renamed to match common Python/CLI naming)
  - `model2petab` -> `model_to_petab`
  - `models2petab` -> `models_to_petab`
- Predecessor model ID, to indicate the model that was previous to a model during model selection iterations

## 0.0.2
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
- Produce graphs given a set of models and their predecessor models
- Allow specification of `model_id` in a model space file, if the row describes only a single model?
- Use `predecessor_model` everywhere (e.g. `VIRTUAL_PREDECESSOR_MODEL`, `predecessor_model_files`)
