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

sys.path.append(os.path.join("..", ".."))
sys.path.append(os.path.join("..", "..", "notebooks"))
from common import *

# %% [markdown]
# # Load Experiments

# %%
output_path_base = os.path.join("..", "..", "output")

# Replica prefixes for collecting the experiments.
replica_prefixes = {
    "SBST": ["SBST"]
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
print(experiments)

