import math
from job import Job
import unittest

def myfunc(input):
    x1, x2, x3 = input[0], input[1], input[2]
    h1 = 305 - 100 * (math.sin(x1 / 3) + math.sin(x2 / 3) + math.sin(x3 / 3))
    h2 = 230 - 75 * (math.cos(x1 / 2.5 + 15) + math.cos(x2 / 2.5 + 15) + math.cos(x3 / 2.5 + 15))
    h3 = (x1 - 7) ** 2 + (x2 - 7) ** 2 + (x3 - 7) ** 2 - (
            math.cos((x1 - 7) / 2.75) + math.cos((x2 - 7) / 2.75) + math.cos((x3 - 7) / 2.75))

    return [h1, h2, h3]


description = {
    "sut": "python.PythonFunction",
    "sut_parameters": {
        "input_range": [[-15, 15], [-15, 15], [-15, 15]],
        "output_range": [[0, 350], [0, 350], [0, 350]],
        "function": myfunc
    },
    "objective_func": ["ObjectiveMaxSelected", "ObjectiveMaxSelected", "ObjectiveMaxSelected"],
    "objective_func_parameters": [
        {"selected": [0], "invert": False, "scale": True},
        {"selected": [1], "invert": False, "scale": True},
        {"selected": [2], "invert": False, "scale": True}],
    "objective_selector": "ObjectiveSelectorMAB",
    "objective_selector_parameters": {"warm_up": 30},
    "algorithm": "platypus.PlatypusOpt",
    "job_parameters": {"N_tests": 80, "N_random_init": 20, "mode": "stop_at_first_falsification"}
}

Job(description).start()
