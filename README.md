<img src="https://raw.githubusercontent.com/PEtab-dev/petab_select/refs/heads/main/doc/logo/logo-wide.svg" height="200" alt="PEtab Select logo">

[![PyPI - Version](https://img.shields.io/pypi/v/petab-select)](https://pypi.org/project/petab-select/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14183390.svg)](https://doi.org/10.5281/zenodo.14183390)


The [PEtab](https://petab.readthedocs.io/) extension for model selection,
including the additional file formats and package.

## Install

The Python 3 library provides both the Python 3 and command-line (CLI)
interfaces, and can be installed from PyPI, with `pip3 install petab-select`.

## Documentation

Further documentation is available at
[http://petab-select.readthedocs.io/](http://petab-select.readthedocs.io/).

## Examples

There are example Jupyter notebooks covering visualization, custom non-SBML
models, and the CLI and Python API, in the `doc/examples` directory.
The notebooks can be viewed at [https://petab-select.readthedocs.io/en/stable/examples.html](https://petab-select.readthedocs.io/en/stable/examples.html).

## Supported features

PEtab Select offers various methods and criteria for model selection, as well
as a variety of visualization options.

### Criteria

- `AIC`: [Akaike information criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion#Definition)
- `AICc`: [Corrected Akaike information criterion](https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size)
- `BIC`: [Bayesian information criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion#Definition)

### Methods

- `forward`:
  [Forward selection](https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches).
  Iteratively increase model complexity.
- `backward`: [Backward selection](https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches).
  Iteratively decrease model complexity.
- `brute_force`. Calibrate all models.
- `famos`:
  [Flexible and dynamic Algorithm for Model Selection (FAMoS)](https://doi.org/10.1371/journal.pcbi.1007230)

Note that the directional methods (forward, backward) find models with the
smallest step size (in terms of number of estimated parameters). For example,
given the forward method and a predecessor model with 2 estimated parameters,
if there are no models with 3 estimated parameters, but some models with 4
estimated parameters, then the search may return candidate models with 4
estimated parameters.
