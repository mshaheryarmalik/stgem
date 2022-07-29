#!/bin/sh

export PYTHONPATH=`pwd`

# Create .rst files for all py. files in project
sphinx-apidoc -o docs/source . []

# Build html files using the rst. files
sphinx-build -a -b html -c docs/source . docs/build/html

#After running run-sphinx.sh:
#Open following file in web browser to review documentation: docs/build/html/docs/source/index.html