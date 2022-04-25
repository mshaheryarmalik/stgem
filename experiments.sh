#!/bin/bash

export PYTHONPATH=`pwd`
python3 problems/arch-comp-2021/afc/afc.py AFC27 25 25321
echo 'Done: AFC27' >> experiments.log
python3 problems/arch-comp-2021/afc/afc.py AFC29 25 25321
echo 'Done: AFC29' >> experiments.log

python3 problems/arch-comp-2021/at/at.py AT1 25 25321
echo 'Done: AT1' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT2 25 25321
echo 'Done: AT2' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT51 25 25321
echo 'Done: AT51' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT52 25 25321
echo 'Done: AT52' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT53 25 25321
echo 'Done: AT53' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT54 25 25321
echo 'Done: AT54' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT6A 25 25321
echo 'Done: AT6A' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT6B 25 25321
echo 'Done: AT6B' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT6C 25 25321
echo 'Done: AT6C' >> experiments.log
python3 problems/arch-comp-2021/at/at.py AT6ABC 25 25321
echo 'Done: AT6ABC' >> experiments.log

python3 problems/arch-comp-2021/f16/f16.py F16 25 25321
echo 'Done: F16' >> experiments.log

python3 problems/arch-comp-2021/nn/nn.py NN 25 25321
echo 'Done: NN' >> experiments.log
python3 problems/arch-comp-2021/nn/nn.py NNX 25 25321
echo 'Done: NNX' >> experiments.log

python3 problems/arch-comp-2021/sc/sc.py SC 25 25321
echo 'Done: SC' >> experiments.log

