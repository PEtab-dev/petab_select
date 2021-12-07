# Changes
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
