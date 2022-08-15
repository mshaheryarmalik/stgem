#!/bin/sh

export PYTHONPATH=`pwd`

cd tests-parser
pytest -s test_signal.py
#pytest -s test_interval.py