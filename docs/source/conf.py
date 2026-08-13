"""Sphinx configuration for the Q1K/SyneQxis documentation."""

from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version

sys.path.insert(0, os.path.abspath("../.."))

project = "SyneQxis Q1K Pipeline"
author = "Q1K/SyneQxis contributors"
copyright = "2026, Q1K/SyneQxis contributors"

try:
    release = package_version("q1k")
except PackageNotFoundError:  # pragma: no cover - docs build fallback
    release = "0.1.0"
version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.intersphinx",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "numpydoc",
    "sphinx_copybutton",
    "sphinx_design",
]

autosummary_generate = True
autodoc_typehints = "description"
numpydoc_show_class_members = False
numpydoc_class_members_toctree = False

templates_path = ["_templates"]
exclude_patterns = []

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "mne": ("https://mne.tools/stable", None),
    "mne_bids": ("https://mne.tools/mne-bids/stable", None),
    "pylossless": ("https://pylossless.readthedocs.io/en/latest/", None),
}

html_theme = "shibuya"
html_title = "SyneQxis Q1K Pipeline"
html_static_path = []
