#!/bin/sh

export PYTHONPATH=`pwd`

# Remove previous build
cd docs
make clean
cd ..

# Create .rst files for whole project
SPHINX_APIDOC_OPTIONS=noindex,members,show-inheritance # Automodule options
export SPHINX_APIDOC_OPTIONS
sphinx-apidoc -f -o docs/source -T . 'stgem.py' # exclude stgem.py to stop generation of duplicate stgem.rst files

# Build html files using the rst. files
sphinx-build -a -b html -c docs/source . docs/build/html

# After running run-sphinx.sh:
# Open following file in web browser to review documentation: docs/build/html/docs/source/index.html