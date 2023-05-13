# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'RLib'
copyright = '2023, Axel Dinh Van Chi'
author = 'Axel Dinh Van Chi'
release = '0.0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.autodoc",
              "sphinx.ext.mathjax"]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'furo'
html_static_path = ['_static']

import pathlib
import sys
print(pathlib.Path(__file__).parents[2].resolve().joinpath("src").as_posix())
sys.path.insert(0, pathlib.Path(__file__).parents[2].resolve().joinpath("src/rlib").as_posix())
