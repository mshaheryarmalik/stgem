#!/bin/sh

# Run the experiments for our ARCH-COMP paper.

cd ..
export PYTHONPATH=`pwd`
cd problems/arch-comp-2021

python3 run.py AT AT1 '' 50 25321 ARCH_OGAN
##python3 run.py AT AT2 '' 50 25321 ARCH_OGAN
##python3 run.py AT AT51 '' 50 25321 ARCH_OGAN
##python3 run.py AT AT52 '' 50 25321 ARCH_OGAN
##python3 run.py AT AT53 '' 50 25321 ARCH_OGAN
##python3 run.py AT AT54 '' 50 25321 ARCH_OGAN
python3 run.py AT AT6A '' 50 25321 ARCH_OGAN
python3 run.py AT AT6B '' 50 25321 ARCH_OGAN
python3 run.py AT AT6C '' 50 25321 ARCH_OGAN
##python3 run.py AT ATX13 '' 50 25321 ARCH_OGAN
##python3 run.py AT ATX14 '' 50 25321 ARCH_OGAN
python3 run.py AT ATX2 '' 50 25321 ARCH_OGAN
python3 run.py AT ATX61 '' 50 25321 ARCH_OGAN
python3 run.py AT ATX62 '' 50 25321 ARCH_OGAN

python3 run.py F16 F16 '' 50 25321 ARCH_OGAN

python3 run.py AFC AFC27 normal 50 25321 ARCH_OGAN
##python3 run.py AFC AFC29 normal 50 25321 ARCH_OGAN

python3 run.py NN NN normal 50 25321 NN_WOGAN

##python3 run.py CC CC1 normal 50 25321 ARCH_OGAN
##python3 run.py CC CC2 normal 50 25321 ARCH_OGAN
python3 run.py CC CC3 normal 50 25321 ARCH_OGAN
python3 run.py CC CC4 normal 50 25321 ARCH_OGAN
##python3 run.py CC CC5 normal 50 25321 ARCH_OGAN

##python3 run.py SC SC normal 50 25321 ARCH_OGAN
