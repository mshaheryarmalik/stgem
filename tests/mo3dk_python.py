import math
from stgem.job import Job
import unittest

def myfunc(input:  [[-15, 15], [-15, 15], [-15, 15]] ) -> [ [0, 350], [0, 350], [0, 350]]:
    x1, x2, x3 = input[0], input[1], input[2]
    h1 = 305 - 100 * (math.sin(x1 / 3) + math.sin(x2 / 3) + math.sin(x3 / 3))
    h2 = 230 - 75 * (math.cos(x1 / 2.5 + 15) + math.cos(x2 / 2.5 + 15) + math.cos(x3 / 2.5 + 15))
    h3 = (x1 - 7) ** 2 + (x2 - 7) ** 2 + (x3 - 7) ** 2 - (
            math.cos((x1 - 7) / 2.75) + math.cos((x2 - 7) / 2.75) + math.cos((x3 - 7) / 2.75))

    # we make it easy to falsify
    return [h1, h2, h3-20]

description = {
    "sut": "python.PythonFunction",
    "sut_parameters": {"function": myfunc},
    "objective_func": ["ObjectiveMinSelected", "ObjectiveMinSelected", "ObjectiveMinSelected"],
    "objective_func_parameters": [
        {"selected": [0], "invert": False, "scale": True},
        {"selected": [1], "invert": False, "scale": True},
        {"selected": [2], "invert": False, "scale": True}],
    "objective_selector": "ObjectiveSelectorMAB",
    "objective_selector_parameters": {"warm_up": 30},
    "algorithm": "ogan.OGAN",
    "algorithm_parameters":
        {"input_dimension": 3,
         "use_predefined_random_data": False,
         "predefined_random_data": {"test_inputs": None, "test_outputs": None},
         "fitness_coef": 0.95,
         "train_delay": 0,
         "N_candidate_tests": 1,
         "ogan_model": "model_keras.OGANK_Model",
         "ogan_model_parameters": {
             "optimizer": "Adam",
             "d_epochs": 10,
             "noise_bs": 10000,
             "g_epochs": 1,
             "d_size": 512,
             "g_size": 512,
             "d_adam_lr": 0.001,
             "g_adam_lr": 0.0001,
             "noise_dimensions": 50,
             "noise_batch_size": 10000
         },
         "train_settings_init": {"epochs": 1, "discriminator_epochs": 10, "generator_epochs": 1},
         "train_settings": {"epochs": 1, "discriminator_epochs": 10, "generator_epochs": 1}
         },
    "job_parameters": {"N_tests": 80, "N_random_init": 20, "mode": "stop_at_first_falsification"}
}

r=Job(description).start()
r.dump_to_file("mo3k_python_results.pickle")
