#!/bin/sh

export PYTHONPATH=`pwd`
cd tests-matlab
python3 -m pytest

