#!/bin/sh

export PYTHONPATH=`pwd`
cd tests-matlab
python3 ../tests/test_jobs.py

