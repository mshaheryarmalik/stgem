#!/bin/sh

export PYTHONPATH=`pwd`

cd docs
make clean
cd ..

# Create .rst files for whole project
sphinx-apidoc -f -o docs/source -T . []

# Build html files using the rst. files
sphinx-build -a -b html -c docs/source . docs/build/html

#After running run-sphinx.sh:
#Open following file in web browser to review documentation: docs/build/html/docs/source/index.html