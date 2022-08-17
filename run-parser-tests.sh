#!/bin/sh

export PYTHONPATH=`pwd`

cd tests-parser

#pytest -s test_interval.py # Does not run currently
pytest -s test_phi.py
pytest -s test_signal.py