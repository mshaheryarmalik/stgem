import os, sys

import stl.robustness as STL


from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS
from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model
from stgem.generator import Search
from stgem.objective_selector import ObjectiveSelectorAll
from stgem.objective import FalsifySTL
from stgem.sut.matlab.sut import Matlab

sys.path.append(os.path.split(os.path.dirname(__file__))[-1])
from f16_python_sut import F16GCAS_PYTHON2, F16GCAS_PYTHON3

ogan_parameters = {"fitness_coef": 0.95,
                   "train_delay": 1,
                   "N_candidate_tests": 1,
                   "reset_each_training": True
                   }

ogan_model_parameters = {
    "dense": {
        "optimizer": "Adam",
        "discriminator_lr": 0.005,
        "discriminator_betas": [0.9, 0.999],
        "generator_lr": 0.0005,
        "generator_betas": [0.9, 0.999],
        "noise_batch_size": 2048,
        "generator_loss": "MSE,Logit",
        "discriminator_loss": "MSE,Logit",
        "generator_mlm": "GeneratorNetwork",
        "generator_mlm_parameters": {
            "noise_dim": 20,
            "neurons": 64
        },
        "discriminator_mlm": "DiscriminatorNetwork",
        "discriminator_mlm_parameters": {
            "neurons": 64,
            "discriminator_output_activation": "sigmoid"
        },
        "train_settings_init": {"epochs": 2,
                                "discriminator_epochs": 20,
                                "generator_batch_size": 32},
        "train_settings": {"epochs": 1,
                           "discriminator_epochs": 30,
                           "generator_batch_size": 32}
    }
}

def build_specification(selected_specification, mode=None, asut=None):
    from math import pi

    # ARCH-COMP
    roll_range = [0.2*pi, 0.2833*pi]
    pitch_range = [-0.4*pi, -0.35*pi]
    yaw_range = [-0.375*pi, -0.125*pi]
    # PART-X
    """
    roll_range = [0.2*pi, 0.2833*pi]
    pitch_range = [-0.5*pi, -0.54*pi]
    yaw_range = [0.25*pi, 0.375*pi]
    """
    # FULL
    """
    roll_range = [-pi, pi]
    pitch_range = [-pi, pi]
    yaw_range = [-pi, pi]
    """

    sut_parameters = {"model_file": "f16/run_f16",
                      "init_model_file": "f16/init_f16",
                      "input_type": "vector",
                      "output_type": "signal",
                      "inputs": ["ROLL", "PITCH", "YAW"],
                      "outputs": ["ALTITUDE"],
                      "input_range": [roll_range, pitch_range, yaw_range],
                      "output_range": [[0, 4040]], # Starting altitude defined in init_f16.m.
                      "initial_altitude": 4040, # Used by the Python SUTs.
                      "simulation_time": 15
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab(sut_parameters)
        #asut = F16GCAS_PYTHON2(sut_parameters)
        #asut = F16GCAS_PYTHON3(sut_parameters)

    # Notice that here the input is a vector.
    scale = True
    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "F16":
        specification = "always[0,15] ALTITUDE > 0"
        specifications = [specification]
        strict_horizon_check = True
        epsilon = 0.0
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return asut, specifications, sut_parameters, scale, strict_horizon_check, epsilon

def objective_selector_factory():
    return ObjectiveSelectorAll()

def get_objective_selector_factory():
    return objective_selector_factory

def step_factory():
    mode = "stop_at_first_objective"

    step_1 = Search(mode=mode,
                    budget_threshold={"executions": 75},
                    #algorithm=Random(model_factory=(lambda: LHS(parameters={"samples": 75})))
                    algorithm=Random(model_factory=(lambda: Uniform()))
                   )      
    step_2 = Search(mode=mode,
                    budget_threshold={"executions": 300},
                    #algorithm=WOGAN(model_factory=(lambda: WOGAN_Model()))
                    #algorithm=OGAN(model_factory=(lambda: OGANK_Model()))
                    algorithm=OGAN(model_factory=(lambda: OGAN_Model(ogan_model_parameters["dense"])), parameters=ogan_parameters),
                    results_include_models=False
                   )

    #steps = [step_1]
    steps = [step_1, step_2]
    return steps

def get_step_factory():
    return step_factory

