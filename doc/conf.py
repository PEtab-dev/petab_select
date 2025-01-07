# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
from __future__ import annotations

import inspect

import sphinx

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "PEtab Select"
copyright = "2024, The PEtab Select developers"
author = "The PEtab Select developers"


# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "readthedocs_ext.readthedocs",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx.ext.mathjax",
    "nbsphinx",
    "IPython.sphinxext.ipython_console_highlighting",
    "recommonmark",
    "sphinx_autodoc_typehints",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


intersphinx_mapping = {
    "petab": (
        "https://petab.readthedocs.io/projects/libpetab-python/en/latest/",
        None,
    ),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "numpy": ("https://numpy.org/devdocs/", None),
    "python": ("https://docs.python.org/3", None),
}

autosummary_generate = True
autodoc_default_options = {
    "special-members": "__init__",
    "inherited-members": True,
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["standard"]
html_logo = "logo/logo-wide.svg"


def autodoc_skip_member(app, what, name, obj, skip, options):
    """Exclude some objects from the documentation."""
    if inspect.isbuiltin(obj):
        return True

    # Skip inherited members from builtins
    #  (skips, for example, all the int/str-derived methods of enums
    if (
        objclass := getattr(obj, "__objclass__", None)
    ) and objclass.__module__ == "builtins":
        return True

    return None


def setup(app: sphinx.application.Sphinx):
    app.connect("autodoc-skip-member", autodoc_skip_member, priority=0)
