#!/bin/sh

export PYTHONPATH=`pwd`
cd tests
pytest test_load.py

