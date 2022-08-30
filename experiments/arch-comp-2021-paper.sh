#!/bin/sh

# Run the experiments of Table 2 of "Falsification of Hybrid Systems Using
# Adaptive Probabilistic Search".

cd ..
export PYTHONPATH=`pwd`
cd problems/arch-comp-2021

python3 run.py AT AT1 '' 50 25321 AT1_ALVTS
python3 run.py AT AT2 '' 50 25321 AT2_ALVTS
##python3 run.py AT AT51 '' 50 25321 AT51_ALVTS
##python3 run.py AT AT52 '' 50 25321 AT52_ALVTS
##python3 run.py AT AT53 '' 50 25321 AT53_ALVTS
##python3 run.py AT AT54 '' 50 25321 AT54_ALVTS
python3 run.py AT AT6A '' 50 25321 AT6A_ALVTS
python3 run.py AT AT6B '' 50 25321 AT6B_ALVTS
python3 run.py AT AT6C '' 50 25321 AT6C_ALVTS
python3 run.py AT ATX13 '' 50 25321 ATX13_ALVTS
python3 run.py AT ATX14 '' 50 25321 ATX14_ALVTS
python3 run.py AT ATX2 '' 50 25321 ATX2_ALVTS
python3 run.py AT ATX61 '' 50 25321 ATX61_ALVTS
python3 run.py AT ATX62 '' 50 25321 ATX62_ALVTS

python3 run.py F16 F16 '' 50 25321 F16_ALVTS

python3 run.py AFC AFC27 normal 50 25321 AFC27_ALVTS
##python3 run.py AFC AFC29 normal 50 25321 AFC29_ALVTS

##python3 run.py CC CC1 normal 50 25321 CC1_ALVTS
##python3 run.py CC CC2 normal 50 25321 CC2_ALVTS
##python3 run.py CC CC3 normal 50 25321 CC3_ALVTS
python3 run.py CC CC4 normal 50 25321 CC4_ALVTS
##python3 run.py CC CC5 normal 50 25321 CC5_ALVTS

python3 run.py NN NN normal 50 25321 NN_ALVTS
#python3 run.py NN NNX normal 50 25321 NNX_ALVTS

