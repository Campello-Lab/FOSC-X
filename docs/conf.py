# -- Path setup --------------------------------------------------------------

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# -- Project information -----------------------------------------------------

project = 'FOSC-X'
copyright = '2026, Connor Simpson'
author = 'Connor Simpson'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
primary_domain = 'py'
default_role = 'py:obj'

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
]
napoleon_google_docstring = False
napoleon_numpy_docstring = True

# Markdown support
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# Ignore build artifacts
exclude_patterns = ['_build', 'build', '**.ipynb_checkpoints']

templates_path = ['_templates']

# -- HTML output -------------------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# -- Autodoc settings --------------------------------------------------------

autodoc_member_order = 'bysource'
autoclass_content = 'both'

autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'inherited-members': False,
}

autodoc_typehints = "description"

autodoc_mock_imports = []

# Skip sklearn internals
def skip_member(app, what, name, obj, skip, options):
    if getattr(obj, "__module__", "").startswith("sklearn"):
        return True
    return skip

# -- Image generation hook ---------------------------------------------------

def setup(app):
    app.connect("autodoc-skip-member", skip_member)
