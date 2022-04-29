# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.3.4
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---
%matplotlib inline
import os, sys

sys.path.append(os.path.join(".."))
from common import *

# %%
output_path_base = os.path.join("..", "..", "..", "output")

specification = "AT1"
experiments = ["AT"]

results = {}
for experiment in experiments:
    r = load(os.path.join(output_path_base, experiment), specification + "_")
    if len(r) > 0:
        results[experiment] = r

# %%
print("Falsification rates:")
for specification in results:
    print("{}: {}".format(specification, falsification_rate(results[specification])))

# %%
replica = 0
result = results["AT"][replica]

# %%
plotTest(result, 100)

# %%
anim = animateResult(result)
HTML(anim.to_jshtml())

