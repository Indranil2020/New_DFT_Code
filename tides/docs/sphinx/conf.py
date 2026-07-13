# TIDES Sphinx configuration.
#
# Theory manual and API documentation for the TIDES DFT engine.
# Build with: make html

import os
import sys

# Add the Python API source root so autodoc can import tides.
sys.path.insert(0, os.path.abspath("../../api/python"))

# -- Project information -----------------------------------------------------

project = "TIDES"
copyright = "2026, TIDES Contributors"
author = "TIDES Contributors"
release = "0.1.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.mathjax",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinxcontrib.bibtex",
    "recommonmark",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]

# -- MathJax configuration ---------------------------------------------------

mathjax_path = (
    "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
)
mathjax_options = {
    "async": "async",
}

# -- Intersphinx configuration -----------------------------------------------

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}

# -- Autodoc configuration ---------------------------------------------------

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
}

# -- BibTeX configuration ----------------------------------------------------

bibtex_bibfiles = ["bibliography.bib"]
