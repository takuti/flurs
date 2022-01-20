# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))

import inspect
import os
import subprocess
import sys
from datetime import datetime

import pkg_resources

GH_ORGANIZATION = "takuti"
GH_PROJECT = "flurs"
PACKAGE = "flurs"


def linkcode_resolve(domain, info):
    """Generate link to GitHub.
    References:
    - https://github.com/scikit-learn/scikit-learn/blob/f0faaee45762d0a5c75dcf3d487c118b10e1a5a8/doc/conf.py
    - https://github.com/chainer/chainer/pull/2758/
    """
    if domain != "py" or not info["module"]:
        return None

    # tag
    try:
        revision = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()
    except (subprocess.CalledProcessError, OSError):
        print("Failed to execute git to get revision")
        return None
    revision = revision.decode("utf-8")

    obj = sys.modules.get(info["module"])
    if obj is None:
        return None
    for comp in info["fullname"].split("."):
        obj = getattr(obj, comp, None)

    # filename
    try:
        filename = inspect.getsourcefile(obj)
    except Exception:
        return None
    if filename is None:
        return None

    # relpath
    pkg_root_dir = os.path.dirname(__import__(PACKAGE).__file__)
    filename = os.path.realpath(filename)
    if not filename.startswith(pkg_root_dir):
        return None
    relpath = os.path.relpath(filename, pkg_root_dir)

    # line number
    try:
        linenum = inspect.getsourcelines(obj)[1]
    except Exception:
        linenum = ""

    return "https://github.com/{}/{}/blob/{}/{}/{}#L{}".format(
        GH_ORGANIZATION, GH_PROJECT, revision, PACKAGE, relpath, linenum
    )


# -- Project information -----------------------------------------------------

project = GH_PROJECT
copyright = f'2017-{datetime.now().year}, Takuya Kitazawa'
author = 'Takuya Kitazawa'

# The full version, including alpha/beta/rc tags
release = pkg_resources.get_distribution(PACKAGE).version


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.doctest",
    "sphinx.ext.linkcode",
    "sphinx.ext.mathjax",
    "sphinx.ext.napoleon",
]

autodoc_default_options = {
    "members": None,
    "member-order": "bysource",
    "special-members": "__init__",
    "undoc-members": True,
    "exclude-members": None,
}

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = '.rst'

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The name of the Pygments (syntax highlighting) style to use.
pygments_style = 'sphinx'

# Output file base name for HTML help builder.
htmlhelp_basename = 'FluRSdoc'

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["images"]
