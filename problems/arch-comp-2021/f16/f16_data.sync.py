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
%matplotlib widget
import os, sys

sys.path.append(os.path.join("..", "..", ".."))

import matplotlib.pyplot as plt
from matplotlib import cm
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

specification = "F16"
experiments = ["F16", "F16_dense", "F16_batch", "F16_300", "F16_ne", "F16_lr1", "F16_uniform", "F16_uniform_keras"]

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
for experiment in experiments:
    print("{}: {}".format(experiment, falsification_rate(results[experiment])))

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
figs, axs = plt.subplots(len(experiments), 1, figsize=(10, 30))
for i, experiment in enumerate(experiments):
    A = mean_min_along(results[experiment], length=300)
    axs[i].set_title(experiment)
    axs[i].set_ylim(0.00, 0.08)
    axs[i].plot(A)

# %%
idx = 0
i = 0
X, Z, Y = results_1[idx].test_repository.get()

print("Input:")
print("  ROLL = {}".format(Z[i][0][0]))
print("  PITCH = {}".format(Z[i][0][1]))
print("  YAW = {}".format(Z[i][0][2]))

fig, axs = plt.subplots(1, 1)
fig.suptitle("Output signals:")
axs.set_title("ALTITUDE")
axs.minorticks_on()
axs.plot(Z[i][3], Z[i][1][0])

print("Robustness: {}".format(Y[i]))

# %%
experiment = "F16_300"
#experiment = "F16_uniform"
idx = 5
angle = 25

X, _, Y = data = results[experiment][idx].test_repository.get()
X = np.asarray(X)
Y = np.asarray(Y).reshape(-1)

#X = X[200:250,:]
#Y = Y[200:250]

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot(211, projection="3d")
ax.azim = angle # rotate around the z axis
#ax.dist = 1
#ax.elev = elev

# Map for transforming the objective value to a color.
def transform_objective(x):
    x = max(0, min(0.1, x))
    return 1 - 5*x

def transform_index(idx):
    return (idx - 75)/150

color_map = cm.get_cmap("Reds", 8)
#colors = ["green" if i < 75 else color_map(transform_objective(Y[i])) for i in range(X.shape[0])]
colors = []
for i in range(X.shape[0]):
    if Y[i] <= 0.0:
        colors.append("purple")
    elif i < 75:
        colors.append("grey")
    else:
        colors.append(color_map(transform_index(i)))
ax.scatter(X[:,0], X[:,1], X[:,2], s=50, alpha=0.6, edgecolors="w", c=colors)

ax.set_xlabel("1")
ax.set_xlim(-1, 1)
ax.set_ylabel("2")
ax.set_ylim(-1, 1)
ax.set_zlabel("3")
ax.set_zlim(-1, 1)

ax = fig.add_subplot(212, projection="3d")
ax.azim = angle

A = [i for i in range(X.shape[0]) if Y[i] < 0.03]
colors = ["green" if i < 75 else color_map(transform_index(i)) for i in A]
ax.scatter(X[A,0], X[A,1], X[A,2], s=50, alpha=0.6, edgecolors="w", c=colors)

ax.set_xlabel("1")
ax.set_xlim(-1, 1)
ax.set_ylabel("2")
ax.set_ylim(-1, 1)
ax.set_zlabel("3")
ax.set_zlim(-1, 1)

plt.show()
