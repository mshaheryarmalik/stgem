#!/bin/sh

# Run the experiments for the extended SBST and WOGAN paper.

cd ..
export PYTHONPATH=`pwd`
cd problems/sbst

python3 run.py 20 OLD
python3 run.py 20 DAVE2_OLD
python3 run.py 20 NEW_DISTANCE
python3 run.py 20 DAVE2_NEW_DISTANCE
