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
from math import degrees

sys.path.append(os.path.join("..", ".."))
sys.path.append(os.path.join("..", "..", "notebooks"))
from common import *

# %% [markdown]
# # Domain-Specific Code

# %%
def steering_sd(sut_output):
    """Computes the standard deviation of the steering angle from a SUTOutput
    object. This is a behavioral diversity measure used in the SBST 2022
    report."""

    return np.std(sut_output.outputs[3])

def direction_coverage(sut_input, n_bins=36):
    """Compute the coverage of the the road directions. That is, compute the
    angles between two consecutive road points and place the angles into bins
    (default 36 bins, i.e., bins cover 10 degrees) and return the proportion of
    bins covered. The angle is defined as the angle to the vertical axis."""

    points = sut_input.input_denormalized

    # Compute the angles.
    angles = []
    for i in range(0, points.shape[1], 2):
        vector = np.array([points[0,i+1] - points[0,i], points[1,i+1] - points[1,i]])
        angle = degrees(np.arccos( vector[1] / np.linalg.norm(vector) ))
        angles.append(angle)

    # Place into bins.
    bins = np.linspace(0.0, 360.0, num=n_bins + 1)
    covered_bins = set(np.digitize(angles, bins))

    return len(covered_bins) / len(bins)

# %% [markdown]
# # Load Experiments

# %%
output_path_base = os.path.join("..", "..", "output")

# Replica prefixes for collecting the experiments.
replica_prefixes = {
    "SBST": ["SBST", "1000"]
}

experiments = loadExperiments(output_path_base, ["SBST"], replica_prefixes)
experiments = experiments["SBST"]

failure_threshold = 0.05

# %% [markdown]
# # Number of Failed Tests

# %%
def number_of_failed_tests(experiment, threshold):
    out = []
    for result in experiment:
        _, _, Y = result.test_repository.get()
        Y = np.array(Y).reshape(-1)
        N = sum(y <= threshold for y in Y)
        out.append(N)

    return out

# %%
data = []
for identifier in experiments:
    failed_tests = number_of_failed_tests(experiments[identifier], failure_threshold)
    data.append(failed_tests)
    print(failed_tests)

own_boxplot(data, list(experiments), title="Number of Failed Tests", ylabel="Number of Failed Tests")

# %%
result = experiments["1000"][0]
tr = result.test_repository
X, Z, Y = tr.get()
Y = np.array(Y).reshape(-1)

idx = []
for i in range(len(Y)):
    if Y[i] <= 0.05:
        idx.append(i)

for i in idx:
    sut_output = Z[i]
    dl = sut_output.outputs[1]
    dr = sut_output.outputs[2]
    j = np.argmin(dl)
    k = np.argmin(dr)
    print(i)
    print("  L: {}, {}, {}".format(j, dl[j], dr[j]))
    print("  R: {}, {}, {}".format(k, dl[k], dr[k]))

