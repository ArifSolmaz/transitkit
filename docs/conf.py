import os
import sys
sys.path.insert(0, os.path.abspath("../src"))

# Project info
project = "TransitKit"
copyright = "2025, Arif Solmaz"
author = "Arif Solmaz"
release = "2.0.0"

# Extensions
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "nbsphinx",
]

# Theme
html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "titles_only": False,
}

# MyST settings
myst_enable_extensions = [
    "colon_fence",
    "deflist",
]

# nbsphinx settings
nbsphinx_execute = "never"  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# Intersphinx
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "astropy": ("https://docs.astropy.org/en/stable/", None),
}

# Source suffix
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

# Exclude patterns
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "**.ipynb_checkpoints"]
