<img src="https://raw.githubusercontent.com/PEtab-dev/petab_select/refs/heads/main/doc/logo/logo-wide.svg" height="200" alt="PEtab Select logo">

[![PyPI - Version](https://img.shields.io/pypi/v/petab-select)](https://pypi.org/project/petab-select/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14183390.svg)](https://doi.org/10.5281/zenodo.14183390)


The repository for the development of the extension to PEtab for model
selection, including the additional file formats and Python 3 package.

## Install

The Python 3 package provides both the Python 3 and command-line (CLI)
interfaces, and can be installed from PyPI, with `pip3 install petab-select`.

## Documentation

Further documentation is available at
[http://petab-select.readthedocs.io/](http://petab-select.readthedocs.io/).

## Examples

There are example Jupyter notebooks for usage of PEtab Select with

- the command-line interface, and
- the Python 3 interface,

in the `doc/examples` directory.

## Supported features

### Criterion

- `AIC`: https://en.wikipedia.org/wiki/Akaike_information_criterion#Definition
- `AICc`: https://en.wikipedia.org/wiki/Akaike_information_criterion#Modification_for_small_sample_size
- `BIC`: https://en.wikipedia.org/wiki/Bayesian_information_criterion#Definition

### Methods

- `forward`: https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches
- `backward`: https://en.wikipedia.org/wiki/Stepwise_regression#Main_approaches
- `brute_force`: Optimize all possible model candidates, then return the model
  with the best criterion value.
- `famos`: https://doi.org/10.1371/journal.pcbi.1007230

Note that the directional methods (forward, backward) find models with the
smallest step size (in terms of number of estimated parameters). For example,
given the forward method and a predecessor model with 2 estimated parameters,
if there are no models with 3 estimated parameters, but some models with 4
estimated parameters, then the search may return candidate models with 4
estimated parameters.
