import os
import sys

sys.path.insert(0, os.path.abspath('..'))

project = 'TransitKit'
copyright = '2024, Arif Solmaz'
author = 'Arif Solmaz'

extensions = [
    'myst_parser',
]

html_theme = 'sphinx_rtd_theme'
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']