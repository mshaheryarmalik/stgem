import os, sys

import tltk_mtl as STL

from stgem.generator import STGEM, Search, run_multiple_generators
from stgem.generator import STGEM, Search
from stgem.budget import Budget
from stgem.sut.matlab.sut import Matlab
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.model import Uniform, LHS
from stgem.objective import FalsifySTL
from stgem.objective_selector import ObjectiveSelectorAll, ObjectiveSelectorMAB

mode = "stop_at_first_objective"
scale = True
specifications = ["NN", # NN
                  "NNX" # NNX
                 ]
selected_specification = "NN"

def build_specification(selected_specification, asut=None):
    # Notice that this only implements the Instance 2 version of the problem where
    # the input signal is split into exactly 3 segments.

    if selected_specification == "NN":
        ref_input_range = [1, 3]
    elif selected_specification == "NNX":
        ref_input_range = [1.95, 2.05]
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    sut_parameters = {"model_file": "problems/arch-comp-2021/nn/run_neural",
                      "init_model_file": "problems/arch-comp-2021/nn/init_neural",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["REF"],
                      "outputs": ["POS"],
                      "input_range": [ref_input_range],
                      "output_range": [[0, 4]],
                      "simulation_time": 40,
                      "time_slices": [13.33],
                      "sampling_step": 0.5
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab_Simulink(sut_parameters)

    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "NN":
        alpha = 0.005
        beta = 0.03
        # inequality := |POS - REF| > alpha + beta*|REF|
        # We make two copies in order to not share state.
        inequality1 = FalsifySTL.StrictlyGreaterThan(1, 0, beta, alpha, STL.Abs(STL.Subtract(S("POS"), S("REF"))), STL.Abs(S("REF")))
        inequality2 = FalsifySTL.StrictlyGreaterThan(1, 0, beta, alpha, STL.Abs(STL.Subtract(S("POS"), S("REF"))), STL.Abs(S("REF")))
        # always[1,37]( inequality implies (always[0,2]( eventually[0,1] not inequality )) )
        specification = STL.Global(1, 37, STL.Implication(inequality1, STL.Finally(0, 2, STL.Global(0, 1, inequality1))))

        strict_horizon_check = True
    elif selected_specification == "NNX":
        # eventually[0,1](POS > 3.2)
        F1 = STL.Finally(0, 1, FalsifySTL.StrictlyGreaterThan(1, 0, 0, 3.2, S("POS")))
        # eventually[1,1.5]( always[0,0.5](1.75 < POS < 2.25) )
        L = FalsifySTL.StrictlyLessThan(0, 1.75, 1, 0, None, S("POS"))
        R = FalsifySTL.StrictlyLessThan(1, 0, 0, 2.25, S("POS"))
        inequality = STL.And(L, R)
        F2 = STL.Finally(1, 1.5, STL.Global(0, 0.5, inequality))
        # always[2,3](1.825 < POS < 2.175)
        L = FalsifySTL.StrictlyLessThan(0, 1.825, 1, 0, None, S("POS"))
        R = FalsifySTL.StrictlyLessThan(1, 0, 0, 2.175, S("POS"))
        inequality = STL.And(L, R)
        F3 = STL.Global(2, 3, inequality)

        specification = STL.And(F1, STL.And(F2, F3))

        strict_horizon_check = True
    else:
        raise Exception("Unknown specification '{}'.".format(selected_specification))

    return asut, specification, strict_horizon_check

ogan_parameters = {"fitness_coef": 0.95,
                   "train_delay": 1,
                   "N_candidate_tests": 1
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
        "generator_lr": 0.0010,
        "generator_betas": [0.9, 0.999],
        "noise_batch_size": 4096,
        "generator_loss": "MSE",
        "discriminator_loss": "MSE",
        "generator_mlm": "GeneratorNetwork",
        "generator_mlm_parameters": {
            "noise_dim": 50,
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

if __name__ == "__main__":
    selected_specification = sys.argv[1]
    N = int(sys.argv[2])
    init_seed = int(sys.argv[3])

    description = "Neural-network Controller",

    # Build a list of SUTs, Budgets, etc. to be used in factory calls in
    # run_multiple_generators.
    asut = None
    sut_list = []
    budget_list = []
    specification_list = []
    objective_list = []
    objective_selector_list = []
    step_list = []
    for i in range(N):
        asut, specification, strict_horizon_check = build_specification(selected_specification, asut)
        budget = Budget()
        objectives = [FalsifySTL(specification=specification, epsilon=0.01, scale=scale, strict_horizon_check=strict_horizon_check)]
        objective_selector = ObjectiveSelectorAll()
        step_1 = Search(mode=mode,
                        budget_threshold={"executions": 50},
                        algorithm=Random(model_factory=(lambda: LHS(parameters={"samples": 50})))
                       )      
        step_2 = Search(mode=mode,
                        budget_threshold={"executions": 200},
                        #algorithm=WOGAN(model_factory=(lambda: WOGAN_Model()))
                        #algorithm=OGAN(model_factory=(lambda: OGANK_Model()))
                        algorithm=OGAN(model_factory=(lambda: OGAN_Model(ogan_model_parameters["convolution"])), parameters=ogan_parameters)
                       )
        steps = [step_1, step_2]

        sut_list.append(asut)
        budget_list.append(budget)
        objective_list.append(objectives)
        objective_selector_list.append(objective_selector)
        step_list.append(steps)

    def generic_list_factory(lst):
        def factory():
            for x in lst:
                yield x

        g = factory()
        return lambda: next(g)

    def seed_factory():
        def seed_generator():
            c = init_seed
            while True:
                yield c
                c += 1

        g = seed_generator()
        return lambda: next(g)

    seed_factory = seed_factory()
    sut_factory = generic_list_factory(sut_list)
    budget_factory = generic_list_factory(budget_list)
    objective_factory = generic_list_factory(objective_list)
    objective_selector_factory = generic_list_factory(objective_selector_list)
    step_factory = generic_list_factory(step_list)

    def callback(result):
        result.dump_to_file(os.path.join("output", "{}_{}.pickle".format(selected_specification, str(result.timestamp).replace(" ", "_"))))

    r = run_multiple_generators(N,
                                description,
                                seed_factory,
                                sut_factory,
                                budget_factory,
                                objective_factory,
                                objective_selector_factory,
                                step_factory,
                                callback=callback
                               )

