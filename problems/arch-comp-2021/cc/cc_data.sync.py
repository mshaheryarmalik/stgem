# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
#%matplotlib inline
import os, sys

sys.path.append(os.path.join(".."))
from common import *

# %%
output_path_base = os.path.join("..", "..", "..", "output")

specification = "CC1"
experiments = [""]

results = {}
for experiment in experiments:
    r = load(os.path.join(output_path_base, experiment), specification + "_")
    if len(r) > 0:
        results[experiment] = r

# %%
experiment = ""
replica = 0
result = results[experiment][replica]

# %%
plotTest(result, 0)

