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
import os, importlib

import matplotlib.pyplot as plt
import numpy as np

from common import *

# %%
output_base_path = os.path.join("..", "..", "output")

experiments = ["ogan_uniform", "wogan_uniform", "random_uniform"]
benchmarks = ["AFC27", "AFC29", "AT1", "AT2", "AT51", "AT52", "AT53", "AT54", "AT6A", "AT6B", "AT6C", "AT6ABC", "CC1", "CC2", "CC3", "CC4", "CC5", "CCX", "F16", "NN", "NNX"]
#benchmarks = ["CC1", "CC2", "CC3", "CC4", "CC5", "CCX"]

raw_data = {}
# Settings this to false reduces memory usage.
load_sut_output = False
for experiment in experiments:
    raw_data[experiment] = {}
    for benchmark in benchmarks:
        raw_data[experiment][benchmark] = load(os.path.join(output_base_path, experiment), benchmark, load_sut_output=load_sut_output)

# %%
data = raw_data

# %%
# Falsification rates.
FR = {}
for experiment in experiments:
    FR[experiment] = {}
    for benchmark in benchmarks:
        FR[experiment][benchmark] = falsification_rate(data[experiment][benchmark])
        if FR[experiment][benchmark] is None:
            FR[experiment][benchmark] = 0

# %%
# Plot falsification rates.
fig = plt.figure(figsize=(10, 10))
X_axis = np.arange(len(benchmarks))
plt.bar(X_axis - 0.2, [FR["ogan_uniform"][benchmark] for benchmark in benchmarks], 0.2, label="OGAN")
plt.bar(X_axis + 0.0, [FR["wogan_uniform"][benchmark] for benchmark in benchmarks], 0.2, label="WOGAN")
plt.bar(X_axis + 0.2, [FR["random_uniform"][benchmark] for benchmark in benchmarks], 0.2, label="RANDOM")

plt.xticks(X_axis, benchmarks)
plt.xlabel("Benchmarks")
plt.ylabel("Falsification Rates")
plt.title("Falsification Rates for Different Algorithms")
plt.legend()
plt.show()

# %%
# First falsifications.
FF = {}
for experiment in experiments:
    FF[experiment] = {}
    for benchmark in benchmarks:
        X = [first_falsification(data[experiment][benchmark][i]) for i in range(len(data[experiment][benchmark]))]
        FF[experiment][benchmark] = [x for x in X if x is not None]

X1 = [FF["ogan_uniform"][benchmark] for benchmark in benchmarks]
X2 = [FF["wogan_uniform"][benchmark] for benchmark in benchmarks]
X3 = [FF["random_uniform"][benchmark] for benchmark in benchmarks]
condensed_boxplot([X1, X2, X3], benchmarks, ["OGAN", "WOGAN", "RANDOM"], ["blue", "orange", "green"])

