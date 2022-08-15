import stgem.objective.Robustness as STL


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
    "dense": {
        "optimizer": "Adam",
        "discriminator_lr": 0.005,
        "discriminator_betas": [0.9, 0.999],
        "generator_lr": 0.0010,
        "generator_betas": [0.9, 0.999],
        "noise_batch_size": 512,
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
        "train_settings_init": {"epochs": 2,
                                "discriminator_epochs": 20,
                                "generator_batch_size": 32},
        "train_settings": {"epochs": 1,
                           "discriminator_epochs": 30,
                           "generator_batch_size": 32}
    }
}

def build_specification(selected_specification, mode=None, asut=None):
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

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab(sut_parameters)

    scale = True
    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "NN":
        alpha = 0.005
        beta = 0.03
        # inequality1 := |POS - REF| > alpha + beta*|REF|
        # inequality2 := alpha + beta*|REF| > |POS - REF|
        inequality1 = FalsifySTL.GreaterThan(STL.Abs(STL.Subtract(S("POS"), S("REF"))),STL.Sum(STL.Constant(alpha),STL.Mult(STL.Constant(beta),STL.Abs(S("REF")))))
        inequality2 = FalsifySTL.GreaterThan(STL.Sum(STL.Constant(alpha),STL.Mult(STL.Constant(beta),STL.Abs(S("REF")))), STL.Abs(STL.Subtract(S("POS"), S("REF"))))
        # always[1,37]( inequality implies (always[0,2]( eventually[0,1] not inequality )) )
        specification = STL.Global(1, 37, STL.Implication(inequality1, STL.Finally(0, 2, STL.Global(0, 1, inequality2))))

        specifications = [specification]
        strict_horizon_check = True
        epsilon = 0.01
    elif selected_specification == "NNX":
        # eventually[0,1](POS > 3.2)
        F1 = STL.Finally(0, 1, FalsifySTL.GreaterThan(S("POS"),STL.Constant(3.2)))
        # eventually[1,1.5]( always[0,0.5](1.75 < POS < 2.25) )
        L = FalsifySTL.LessThan(STL.Constant(1.75), S("POS"))
        R = FalsifySTL.LessThan(S("POS"),STL.Constant(2.25))
        inequality = STL.And(L, R)
        F2 = STL.Finally(1, 1.5, STL.Global(0, 0.5, inequality))
        # always[2,3](1.825 < POS < 2.175)
        L = FalsifySTL.LessThan(STL.Constant(1.825), S("POS"))
        R = FalsifySTL.LessThan(S("POS"),STL.Constant(2.175))
        inequality = STL.And(L, R)
        F3 = STL.Global(2, 3, inequality)

        conjunctive_specification = STL.And(F1, STL.And(F2, F3))

        #specifications = [conjunctive_specification]
        specifications = [F1, F2, F3]
        strict_horizon_check = True
        epsilon = 0.01
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return asut, specifications, scale, strict_horizon_check, epsilon

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

