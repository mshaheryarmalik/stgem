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
import os, sys

sys.path.append(os.path.join("..", "..", ".."))

import matplotlib.pyplot as plt
import numpy as np

from stgem.generator import STGEMResult

# %%
def load_results(files):
    results = []
    for file in files:
        results.append(STGEMResult.restore_from_file(file))

    return results

# %%
def load(path, prefix):
    files = [os.path.join(path, file) for file in os.listdir(path) if os.path.basename(file).startswith(prefix)]
    files.sort()
    return load_results(files)

output_path_base = os.path.join("..", "..", "..", "output")

specification = "NN"
experiments = ["NN_batch_uniform"]

results = {}
for experiment in experiments:
    r = load(os.path.join(output_path_base, experiment), specification + "_")
    if len(r) > 0:
        results[experiment] = r

# %%
def falsification_rate(results):
    c = 0
    for result in results:
        c += 1 if any(step.success for step in result.step_results) else 0

    return c/len(results)

print("Falsification rates:")
for specification in results:
    print("{}: {}".format(specification, falsification_rate(results[specification])))

# %%
def min_along(X, length=None):
    # Return the minimum so far along X.
    m = 1.0
    A = []
    for i in range(len(X) if length is None else length):
        o = X[i] if i < len(X) else 1.0
        if o < m:
            m = o
        A.append(m)
    return A

def mean_min_along(results, length=None):
    A = []
    for i in range(len(results)):
        _, _, Y = results[i].test_repository.get()
        Y = [Y[i][0] for i in range(len(Y))]
        B = min_along(Y, length=length)
        A.append(B)

    A = np.array(A)
    C = np.mean(A, axis=0)

    return C

# %%
figs, axs = plt.subplots(len(experiments), 1, figsize=(10, 20))
for i, experiment in enumerate(experiments):
    A = mean_min_along(results[experiment], length=300)
    axs.set_title(experiment)
    #axs.set_ylim(0.00, 0.08)
    axs.plot(A)

