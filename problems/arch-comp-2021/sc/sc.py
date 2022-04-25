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
scale = False
specifications = ["SC", # SC
                 ]
selected_specification = "SC"

# Running the model requires Deep Learning Toolbox in Matlab.

def build_specification(selected_specification, asut=None):
    # Notice that this only implements the Instance 2 version of the problem where
    # the input signal is split into exactly 20 segments.

    sut_parameters = {"model_file": "problems/arch-comp-2021/sc/run_steamcondenser",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["FS"],
                      "outputs": ["T", "FCW", "Q", "PRESSURE"],
                      "input_range": [[3.99, 4.01]],
                      "output_range": [None, None, None, [86, 90]],
                      "simulation_time": 35,
                      "time_slices": [1.75],
                      "sampling_step": 0.5
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab_Simulink(sut_parameters)

    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "SC":
        # always[30,35](87 <= pressure <= 87.5)
        L = STL.LessThan(0, 87, 1, 0, None, S("PRESSURE"))
        R = STL.LessThan(1, 0, 0, 87.5, S("PRESSURE"))
        inequality = STL.And(L, R)
        specification = STL.Global(30, 35, inequality)

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
        "generator_loss": "MSE",
        "discriminator_loss": "MSE",
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
        "generator_loss": "MSE,Logit",
        "discriminator_loss": "MSE,Logit",
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

    description = "Steam Condenser with Recurrent Neural Network Controller"

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

