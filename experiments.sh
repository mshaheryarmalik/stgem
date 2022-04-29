#!/bin/bash

export PYTHONPATH=`pwd`
python3 problems/arch-comp-2021/afc/afc.py AFC27_ogan_uniform 25 25321
echo 'Done: AFC27' >> experiments.log
python3 problems/arch-comp-2021/afc/afc.py AFC29_ogan_uniform 25 25321
echo 'Done: AFC29' >> experiments.log

python3 problems/arch-comp-2021/at/at.py AT1_ogan_uniform 25 25321
echo 'Done: AT1' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT2_ogan_uniform 25 25321
echo 'Done: AT2' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT51_ogan_uniform 25 25321
echo 'Done: AT51' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT52_ogan_uniform 25 25321
echo 'Done: AT52' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT53_ogan_uniform 25 25321
echo 'Done: AT53' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT54_ogan_uniform 25 25321
echo 'Done: AT54' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT6A_ogan_uniform 25 25321
echo 'Done: AT6A' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT6B_ogan_uniform 25 25321
echo 'Done: AT6B' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT6C_ogan_uniform 25 25321
echo 'Done: AT6C' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT6ABC_ogan_uniform 25 25321
echo 'Done: AT6ABC' >> experiments.log

python3 problems/arch-comp-2021/f16/f16.py F16_ogan_uniform 25 25321
echo 'Done: F16' >> experiments.log

python3 problems/arch-comp-2021/nn/nn.py NN_ogan_uniform 25 25321
echo 'Done: NN' >> experiments.log
python3 problems/arch-comp-2021/nn/nn.py NNX_ogan_uniform 25 25321
echo 'Done: NNX' >> experiments.log

