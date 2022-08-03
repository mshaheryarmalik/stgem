#!/bin/sh

# Run the experiments of Table 2 of "Falsification of Hybrid Systems Using
# Adaptive Probabilistic Search".

cd ..
export PYTHONPATH=`pwd`
cd problems/arch-comp-2021

python3 run.py AT AT1 '' 50 25321 AT1_ALVTS
python3 run.py AT ATX13 '' 50 25321 ATX13_ALVTS
python3 run.py AT ATX14 '' 50 25321 ATX14_ALVTS
python3 run.py AT ATX2 '' 50 25321 ATX2_ALVTS
python3 run.py AT ATX61 '' 50 25321 ATX61_ALVTS
python3 run.py AT ATX62 '' 50 25321 ATX62_ALVTS

python3 run.py AFC AFC27 normal 50 25321 AFC27_ALVTS

