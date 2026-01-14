# Configuration file for Sphinx documentation

import os
import sys
from datetime import datetime

# Add the source code to the path
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../src'))

# Project information
project = 'TransitKit'
copyright = f'{datetime.now().year}, Arif Solmaz'
author = 'Arif Solmaz'

# Read version from package
try:
    # Try to import directly first
    from transitkit import __version__
    release = __version__
except ImportError:
    # Fall back to reading from file
    try:
        version_file = os.path.join('..', 'src', 'transitkit', '__version__.py')
        with open(version_file) as f:
            exec(f.read())
            release = __version__  # noqa: F821
    except (FileNotFoundError, NameError):
        release = '1.0.0'
version = '.'.join(release.split('.')[:2])

# Extensions
extensions = [
    # Core Sphinx extensions
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.intersphinx',
    'sphinx.ext.mathjax',
    'sphinx.ext.todo',
    'sphinx.ext.githubpages',
    
    # Third-party extensions
    'sphinx_copybutton',
    'sphinx_design',
    'nbsphinx',
    'myst_parser',
    
    # Theme
    'sphinx_rtd_theme',
]

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_use_keyword = True
napoleon_custom_sections = [('Returns', 'params_style')]

# Autodoc settings
autodoc_typehints = 'description'
autodoc_member_order = 'groupwise'
autodoc_default_options = {
    'members': True,
    'member-order': 'groupwise',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__',
    'show-inheritance': True,
    'imported-members': False,
}
autosummary_generate = True
autosummary_generate_overwrite = True

# nbsphinx settings
nbsphinx_execute = 'never'  # Don't execute notebooks on RTD
nbsphinx_prolog = """
{% set docname = 'examples/' + env.doc2path(env.docname, base=None) %}
.. note:: This tutorial is also available as a Jupyter notebook.
   You can `download it from GitHub <https://github.com/arifsolmaz/transitkit/blob/main/{{ docname }}>`_.
"""

# Intersphinx mapping
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'astropy': ('https://docs.astropy.org/en/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'plotly': ('https://plotly.com/python-api-reference/', None),
    'bokeh': ('https://docs.bokeh.org/en/latest/', None),
}

# MyST settings
myst_enable_extensions = [
    'dollarmath',
    'amsmath',
    'deflist',
    'fieldlist',
    'html_admonition',
    'html_image',
    'colon_fence',
    'smartquotes',
    'replacements',
    'linkify',
    'substitution',
]
myst_heading_anchors = 3

# Theme
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': True,
    'style_nav_header_background': '#2980B9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Static files
html_static_path = ['_static']
html_css_files = ['custom.css']
html_js_files = ['custom.js']

# Logo
html_logo = '_static/logo.png'
html_favicon = '_static/favicon.ico'

# Source
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', '**.ipynb_checkpoints']

# Language
language = 'en'

# PDF/LaTeX settings
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '10pt',
    'preamble': r'''
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{graphicx}
    ''',
}

# General
todo_include_todos = True

# Set the master document
master_doc = 'index'

# Smart quotes
smartquotes = True

# If true, keep warnings as "system message" paragraphs in the built documents.
keep_warnings = False

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = True

# Add any paths that contain custom themes here, relative to this directory.
html_theme_path = []