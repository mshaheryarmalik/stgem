import os, sys

import numpy as np

import tltk_mtl as STL

from stgem.generator import STGEM, Search, run_multiple_generators
from stgem.budget import Budget
from stgem.sut.matlab.sut import Matlab
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.random.model import Uniform, LHS
from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model
from stgem.objective import FalsifySTL
from stgem.objective_selector import ObjectiveSelectorAll, ObjectiveSelectorMAB

mode = "stop_at_first_objective"
scale = True
specifications = ["AFC27", # AFC27, normal
                  "AFC29"  # AFC29,AFC33 normal/power
                 ]
selected_specification = "AFC27"

afc_mode = "normal" # normal/power
if afc_mode == "normal":
    throttle_range = [0, 61.2]
elif afc_mode == "power":
    throttle_range = [61.2, 81.2]

def build_specification(selected_specification, asut):
    # Notice that the output MODE is never used in the requirements.
    sut_parameters = {"model_file": "problems/arch-comp-2021/afc/run_powertrain",
                      "init_model_file": "problems/arch-comp-2021/afc/init_powertrain",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["THROTTLE", "ENGINE"],
                      "outputs": ["MU", "MODE"],
                      "input_range": [throttle_range, [900, 1100]],
                      "output_range": [[-1, 1], [0, 1]],
                      "simulation_time": 50,
                      "time_slices": [5, 50],
                      "sampling_step": 0.5
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab_Simulink(sut_parameters)

    # Some ARCH-COMP specifications have requirements whose horizon is longer than
    # the output signal for some reason. Thus strict horizon check needs to be
    # disabled in some cases.
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
        
        strict_horizon_check = False
    elif selected_specification == "AFC29":
        gamma = 0.007
        # always[11,50]( abs(MU) < gamma )
        specification = STL.Global(11, 50, FalsifySTL.StrictlyLessThan(1, 0, 0, gamma, STL.Abs(S("MU"))))
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

    description = "Fuel Control of an Automotive Powertrain ({} mode)".format(afc_mode)

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

