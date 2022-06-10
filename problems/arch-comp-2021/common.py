import os, sys
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import animation
from IPython.display import HTML

sys.path.append(os.path.join("..", ".."))
from stgem.generator import STGEM, STGEMResult
from stgem.budget import Budget

def load_results(files, load_sut_output=True):
    results = []
    for file in files:
        results.append(STGEMResult.restore_from_file(file))

    # This reduces memory usage if these values are not needed.
    if not load_sut_output:
        for result in results:
            result.test_repository._sut_outputs = None

    return results

def load(path, prefix, load_sut_output=True):
    files = [os.path.join(path, file) for file in os.listdir(path) if os.path.basename(file).startswith(prefix)]
    files.sort()

    return load_results(files, load_sut_output)

def falsification_rate(results):
    if len(results) == 0:
        return None

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

def scaleInterval(x, interval_original, interval_target):
    """
    Scales the input value x in interval_original to new interval_target
    """
    A = interval_original[0]
    B = interval_original[1]
    C = interval_target[0]
    D = interval_target[1]
    E = (-1 * A + x) / (-1 * A + B) # input percent of original interval
    return C + E * (-1 * C + D) # input scaled to target interval

def colorIntervals(x, color_map, interval_original, interval_target):
    """
    Uses intervals to return input as a readable color from a color map
    """
    x = scaleInterval(x, interval_original, interval_target)
    return color_map(x)

def color(i, Y, color_map, falsify_pct=0.0, falsify_clr="blue"):

    def transform_objective(x):
        x = max(0, min(0.1, x))
        return 1 + 5 * -x

    if Y[i] <= falsify_pct:
        return falsify_clr
    #elif i < 75:
    #    return "grey"
    else:
        x = Y[i]
        return color_map(transform_objective(x))

def plotTest(result, idx):
    inputs = result.sut_parameters["inputs"]
    outputs = result.sut_parameters["outputs"]
    input_range = result.sut_parameters["input_range"]
    output_range = result.sut_parameters["output_range"]
    simulation_time = result.sut_parameters["simulation_time"]

    X, Z, Y = result.test_repository.get()
    Y = np.array(Y).reshape(-1)

    # Define the signal color based in the test index.
    color_map = cm.get_cmap("Reds", 8) # Use color() to add color to lines if desired

    fig, ax = plt.subplots(2, max(len(inputs), len(outputs)), figsize=(10*max(len(inputs), len(outputs)), 10))

    for i, var in enumerate(inputs):
        ax[0,i].set_title(var)
        ax[0,i].set_xlim((0, simulation_time))
        ax[0,i].set_ylim(input_range[i])
        x = X[idx].input_timestamps
        y = X[idx].input_denormalized[i]
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

    X, Z, Y = result.test_repository.get()
    Y = np.array(Y).reshape(-1)

    # Define the signal color based in the test index.
    color_map = cm.get_cmap("Reds", 8)

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
            x = X[i].input_timestamps
            y = X[i].input_denormalized[j]
            lines[k].set_color(color(i, Y, color_map))
            lines[k].set_data(x, y)
        for j in range(len(outputs)):
            k = len(inputs) + j
            x = Z[i].output_timestamps
            y = Z[i].outputs[j]
            lines[k].set_color(color(i, Y, color_map))
            lines[k].set_data(x, y)

        return lines

    plt.close()
    anim = animation.FuncAnimation(fig, animate, init_func=init, frames=len(Y), interval=200, blit=True)
    return anim

def condensed_boxplot(data, x_labels, pb_labels, colors):
    def set_color(bp, color):
        plt.setp(bp["boxes"], color=color)
        plt.setp(bp["whiskers"], color=color)
        plt.setp(bp["caps"], color=color)
        plt.setp(bp["medians"], color=color)
    
    fig = plt.figure(figsize =(7, 5))
    
    # Create boxplots and set colors.
    for i in range(len(data)):
        #bp = plt.boxplot(data[i])
        bp = plt.boxplot(data[i], positions=i+np.array(range(len(data[i])))*4-0.4, sym="", widths=0.6)
        set_color(bp, colors[i])
    
    # Draw temporary lines and use them to create a legend.
    for i in range(len(pb_labels)):
        plt.plot([], c=colors[i], label=pb_labels[i])
    plt.legend()
    
    plt.xticks(range(0, len(x_labels)*4, 4), x_labels)
    plt.xlim(-2, len(x_labels)*4)
    #plt.ylim(0, 8)
    plt.tight_layout()

