# Configuration file for Sphinx documentation

import os
import sys
from datetime import datetime

# Add the source code to the path
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'TransitKit'
copyright = f'{datetime.now().year}, Your Name'
author = 'Your Name'

# Read version from package
try:
    from transitkit import __version__
    release = __version__
    version = '.'.join(release.split('.')[:2])
except ImportError:
    release = '0.1.0'
    version = '0.1'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'sphinx_copybutton',
    'myst_parser',
    'sphinx_rtd_theme',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True

# Autodoc settings
autodoc_typehints = 'description'
autodoc_member_order = 'bysource'
autodoc_default_options = {
    'members': True,
    'show-inheritance': True,
}

# MyST settings
myst_enable_extensions = [
    'dollarmath',
    'linkify',
    'colon_fence',
]

# Theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
}

# Static files
html_static_path = ['_static']
html_css_files = ['custom.css']

# Source
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Language
language = 'en'