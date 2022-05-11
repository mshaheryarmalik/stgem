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
import numpy as n

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

specifications = ["SC",
                 ]

results = {}
for specification in specifications:
    r = load(os.path.join(output_path_base, "SC"), specification + "_")
    if len(r) > 0:
        results[specification] = r

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
idx = 0
i = 0
X, Z, Y = results_1[idx].test_repository.get()

fig, axs = plt.subplots(1, 1)
fig.suptitle("Input signals:")
axs.set_title("FN")
axs.minorticks_on()
axs.plot(Z[i][2], Z[i][0][0])

fig, axs = plt.subplots(1, 1)
fig.suptitle("Output signals:")
axs.set_title("PRESSURE")
axs.minorticks_on()
axs.plot(Z[i][3], Z[i][1][3])

print("Robustness: {}".format(Y[i]))

