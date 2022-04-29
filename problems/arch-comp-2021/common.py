import os, sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from IPython.display import HTML

sys.path.append(os.path.join("..", "..", ".."))
from stgem.generator import STGEMResult

def load_results(files):
    results = []
    for file in files:
        results.append(STGEMResult.restore_from_file(file))

    return results

def load(path, prefix):
    files = [os.path.join(path, file) for file in os.listdir(path) if os.path.basename(file).startswith(prefix)]
    files.sort()

    return load_results(files)

def falsification_rate(results):
    c = 0
    for result in results:
        c += 1 if any(step.success for step in result.step_results) else 0

    return c/len(results)

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
        Y = np.array(Y).reshape(-1)
        B = min_along(Y, length=length)
        A.append(B)

    A = np.array(A)
    C = np.mean(A, axis=0)

    return C

def first_falsification(result):
    _, _, Y = result.test_repository.get()
    for i in range(len(Y)):
        if min(Y[i]) <= 0.0:
            return i

    return None

def plotTest(result, idx):
    inputs = result.sut_parameters["inputs"]
    outputs = result.sut_parameters["outputs"]
    input_range = result.sut_parameters["input_range"]
    output_range = result.sut_parameters["output_range"]
    simulation_time = result.sut_parameters["simulation_time"]

    _, Z, Y = result.test_repository.get()
    Y = np.array(Y).reshape(-1)

    # Define the signal color based in the test index.
    color_map = cm.get_cmap("Reds", 8)
    def color(i):
        if Y[i] <= 0.0:
            return "blue"
        elif i < 75:
            return "grey"
        else:
            x = Y[i]
            x = max(0, min(0.1, x))
            x = 1 - 5*x
            return color_map(x)

    fig, ax = plt.subplots(2, max(len(inputs), len(outputs)), figsize=(10*max(len(inputs), len(outputs)), 10))

    for i, var in enumerate(inputs):
        ax[0,i].set_title(var)
        ax[0,i].set_xlim((0, simulation_time))
        ax[0,i].set_ylim(input_range[i])
        x = Z[idx].input_timestamps
        y = Z[idx].inputs[i]
        ax[0,i].plot(x, y)
    for i, var in enumerate(outputs):
        ax[1,i].set_title(var)
        ax[1,i].set_xlim((0, simulation_time))
        ax[1,i].set_ylim(output_range[i])
        x = Z[idx].output_timestamps
        y = Z[idx].outputs[i]
        ax[1,i].plot(x, y)

def animateResult(result):
    inputs = result.sut_parameters["inputs"]
    outputs = result.sut_parameters["outputs"]
    input_range = result.sut_parameters["input_range"]
    output_range = result.sut_parameters["output_range"]
    simulation_time = result.sut_parameters["simulation_time"]

    _, Z, Y = result.test_repository.get()
    Y = np.array(Y).reshape(-1)

    # Define the signal color based in the test index.
    color_map = cm.get_cmap("Reds", 8)
    def color(i):
        if Y[i] <= 0.0:
            return "blue"
        elif i < 75:
            return "grey"
        else:
            x = Y[i]
            x = max(0, min(0.1, x))
            x = 1 - 5*x
            return color_map(x)

    # Setup the figures.
    fig, ax = plt.subplots(2, max(len(inputs), len(outputs)), figsize=(10*max(len(inputs), len(outputs)), 10))

    for i, var in enumerate(inputs):
        ax[0,i].set_title(var)
        ax[0,i].set_xlim((0, simulation_time))
        ax[0,i].set_ylim(input_range[i])
    for i, var in enumerate(outputs):
        ax[1,i].set_title(var)
        ax[1,i].set_xlim((0, simulation_time))
        ax[1,i].set_ylim(output_range[i])

    lines = []
    for i in range(len(inputs)):
        lines.append(ax[0,i].plot([], [], lw=2)[0])
    for i in range(len(outputs)):
        lines.append(ax[1,i].plot([], [], lw=2)[0])

    def init():
        for line in lines:
            line.set_data([], [])

        return lines

    def animate(i):
        for j in range(len(inputs)):
            k = j
            x = Z[i].input_timestamps
            y = Z[i].inputs[j]
            lines[k].set_color(color(i))
            lines[k].set_data(x, y)
        for j in range(len(outputs)):
            k = len(inputs) + j
            x = Z[i].output_timestamps
            y = Z[i].outputs[j]
            lines[k].set_color(color(i))
            lines[k].set_data(x, y)

        return lines

    plt.close()
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Y), interval=200, blit=True)
    return anim

