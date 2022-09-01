from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS
from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model
from stgem.generator import Search
from stgem.objective_selector import ObjectiveSelectorAll, ObjectiveSelectorMAB
from stgem.objective import FalsifySTL
from stgem.sut.matlab.sut import Matlab

mode = "stop_at_first_objective"

ogan_parameters = {"fitness_coef": 0.95,
                   "train_delay": 1,
                   "N_candidate_tests": 1,
                   "reset_each_training": True
                   }

ogan_model_parameters = {
    "convolution": {
        "optimizer": "Adam",
        "discriminator_lr": 0.005,
        "discriminator_betas": [0.9, 0.999],
        "generator_lr": 0.0001,
        "generator_betas": [0.9, 0.999],
        "noise_batch_size": 12000,
        "generator_loss": "MSE,Logit",
        "discriminator_loss": "MSE,Logit",
        "generator_mlm": "GeneratorNetwork",
        "generator_mlm_parameters": {
            "noise_dim": 20,
            "hidden_neurons": [64, 64]
        },
        "discriminator_mlm": "DiscriminatorNetwork1dConv",
        "discriminator_mlm_parameters": {
            "feature_maps": [16],
            "kernel_sizes": [[2,2]],
            "convolution_activation": "relu",
            "dense_neurons": 32
        },
        "train_settings_init": {"epochs": 2,
                                "discriminator_epochs": 20,
                                "generator_batch_size": 32},
        "train_settings": {"epochs": 1,
                           "discriminator_epochs": 30,
                           "generator_batch_size": 32}
    }
}

def build_specification(selected_specification, mode=None):
    """Builds a specification object and a SUT for the selected specification.
    In addition, returns if scaling and strict horizon check should be used for
    the specification. A previously created SUT can be passed as an argument,
    and then it will be reused."""

    # Notice that this only implements the Instance 2 version of the problem where
    # the input signal is split into exactly 3 segments.

    if selected_specification == "NN":
        ref_input_range = [1, 3]
    elif selected_specification == "NNX":
        ref_input_range = [1.95, 2.05]
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    sut_parameters = {"model_file": "nn/run_neural",
                      "init_model_file": "nn/init_neural",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["REF"],
                      "outputs": ["POS"],
                      "input_range": [ref_input_range],
                      "output_range": [[0, 4]],
                      "simulation_time": 40,
                      "time_slices": [13.33],
                      "sampling_step": 0.01
                     }

    if selected_specification == "NN":
        alpha = 0.005
        beta = 0.03
        inequality1 = "|POS - REF| > {} + {}*|REF|".format(alpha, beta)
        inequality2 = "{} + {}*|REF| <= |POS - REF|".format(alpha, beta)

        specification = "always[1,37]( {} implies (eventually[0,2]( always[0,1] not {} )) )".format(inequality1, inequality2)

        specifications = [specification]
        strict_horizon_check = True
    elif selected_specification == "NNX":
        F1 = "eventually[0,1](POS > 3.2)"
        F2 = "eventually[1,1.5]( always[0,0.5](1.75 < POS and POS < 2.25) )"
        F3 = "always[2,3](1.825 < POS and POS < 2.175)"

        conjunctive_specification = "{} and {} and {}".format(F1, F2, F3)

        specifications = [conjunctive_specification]
        #specifications = [F1, F2, F3]
        strict_horizon_check = True
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return sut_parameters, specifications, strict_horizon_check

def objective_selector_factory():
    objective_selector = ObjectiveSelectorMAB(warm_up=100)

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
                    algorithm=OGAN(model_factory=(lambda: OGAN_Model(ogan_model_parameters["convolution"])), parameters=ogan_parameters)
                   )
    #steps = [step_1]
    steps = [step_1, step_2]
    return steps

def get_step_factory():
    return step_factory

