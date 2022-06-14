import tltk_mtl as STL

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
        "train_settings_init": {"epochs": 2, "discriminator_epochs": 20, "generator_batch_size": 32},
        "train_settings": {"epochs": 1, "discriminator_epochs": 30, "generator_batch_size": 32}
    },
    "convolution": {
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
        "discriminator_mlm": "DiscriminatorNetwork1dConv",
        "discriminator_mlm_parameters": {
            "feature_maps": [16],
            "kernel_sizes": [[2,2]],
            "convolution_activation": "relu",
            "dense_neurons": 32
        },
        "train_settings_init": {"epochs": 2, "discriminator_epochs": 20, "generator_batch_size": 32},
        "train_settings": {"epochs": 1, "discriminator_epochs": 30, "generator_batch_size": 32}
    }
}

def build_specification(selected_specification, afc_mode="normal", asut=None):
    """Builds a specification object and a SUT for the selected specification.
    In addition, returns if scaling and strict horizon check should be used for
    the specification. A previously created SUT can be passed as an argument,
    and then it will be reused."""

    if afc_mode == "normal":
        throttle_range = [0, 61.2]
    elif afc_mode == "power":
        throttle_range = [61.2, 81.2]

    # Notice that the output MODE is never used in the requirements.
    sut_parameters = {"model_file": "afc/run_powertrain",
                      "init_model_file": "afc/init_powertrain",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["THROTTLE", "ENGINE"],
                      "outputs": ["MU", "MODE"],
                      "input_range": [throttle_range, [900, 1100]],
                      "output_range": [[-1, 1], [0, 1]],
                      "simulation_time": 50,
                      "time_slices": [5, 50],
                      "sampling_step": 0.01
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab(sut_parameters)

    # Some ARCH-COMP specifications have requirements whose horizon is longer than
    # the output signal for some reason. Thus strict horizon check needs to be
    # disabled in some cases.
    scale = True
    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "AFC27":
        beta = 0.008
        # rise := (THROTTLE < 8.8) and (eventually[0,0.05](THROTTLE > 40.0))
        L = FalsifySTL.StrictlyLessThan(1, 0, 0, 8.8, S("THROTTLE"))
        R = STL.Finally(0, 0.05, FalsifySTL.StrictlyGreaterThan(1, 0, 0, 40, S("THROTTLE")))
        rise = STL.And(L, R)
        # fall := (THROTTLE > 40.0) and (eventually[0,0.05](THROTTLE < 8.8))
        L = FalsifySTL.StrictlyGreaterThan(1, 0, 0, 40, S("THROTTLE"))
        R = STL.Finally(0, 0.05, FalsifySTL.StrictlyLessThan(1, 0, 0, 8.8, S("THROTTLE")))
        fall = STL.And(L, R)
        # consequence := always[1,5](abs(MU) < beta)
        consequence = STL.Global(1, 5, FalsifySTL.StrictlyLessThan(1, 0, 0, beta, STL.Abs(S("MU"))))
        # always[11,50]( (rise or fall) implies (consequence)
        specification = STL.Global(11, 50, STL.Implication(STL.Or(rise, fall), consequence))

        specifications = [specification]
        strict_horizon_check = False
        epsilon = 0.01
    elif selected_specification == "AFC29":
        gamma = 0.007
        # always[11,50]( abs(MU) < gamma )
        specification = STL.Global(11, 50, FalsifySTL.StrictlyLessThan(1, 0, 0, gamma, STL.Abs(S("MU"))))

        specifications = [specification]
        strict_horizon_check = True
        epsilon = 0.01
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return asut, specifications, scale, strict_horizon_check, epsilon

def objective_selector_factory():
    return ObjectiveSelectorAll()

def get_objective_selector_factory():
    return objective_selector_factory

def step_factory():
    mode = "stop_at_first_objective"

    step_1 = Search(mode=mode,
                    budget_threshold={"executions": 75},
                    #algorithm=Random(model_factory=(lambda: LHS(parameters={"samples": 50})))
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

