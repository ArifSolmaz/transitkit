import os, sys
sys.path.insert(0, os.path.abspath("../src"))
project = "TransitKit"
copyright = "2024, Arif Solmaz"
author = "Arif Solmaz"
extensions = ["sphinx.ext.autodoc", "sphinx.ext.napoleon", "myst_parser"]
html_theme = "sphinx_rtd_theme"
