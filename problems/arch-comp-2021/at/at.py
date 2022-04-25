import os, sys

import tltk_mtl as STL

from stgem.generator import STGEM, Search, run_multiple_generators
from stgem.budget import Budget
from stgem.sut.matlab.sut import Matlab_Simulink
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.model import Uniform, LHS
from stgem.objective import FalsifySTL
from stgem.objective_selector import ObjectiveSelectorAll, ObjectiveSelectorMAB

mode = "stop_at_first_objective"
scale = True
specifications = ["AT1",
                  "AT2",
                  "AT51",
                  "AT52",
                  "AT53",
                  "AT54",
                  "AT6A",
                  "AT6B",
                  "AT6C",
                  "AT6ABC"
                 ]
selected_specification = "AT1"

def build_specification(selected_specification, asut=None):
    sut_parameters = {"model_file": "problems/arch-comp-2021/at/Autotrans_shift",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["THROTTLE", "BRAKE"],
                      "outputs": ["SPEED", "RPM", "GEAR"],
                      "input_range": [[0, 100], [0, 325]],
                      "output_range": [[0, 200], [0, 7000], [0, 4]],
                      "simulation_time": 30,
                      "time_slices": [5, 5],
                      "sampling_step": 0.2
                     }

    # We allow reusing the SUT for memory conservation (Matlab takes a lot of
    # memory).
    if asut is None:
        asut = Matlab_Simulink(sut_parameters)

    # Some ARCH-COMP specifications have requirements whose horizon is longer than
    # the output signal for some reason. Thus strict horizon check needs to be
    # disabled in some cases.
    S = lambda var: STL.Signal(var, asut.variable_range(var) if scale else None)
    if selected_specification == "AT1":
        # always[0,20](SPEED < 120)
        specification = STL.Global(0, 20, FalsifySTL.StrictlyLessThan(1, 0, 0, 120, S("SPEED")))

        strict_horizon_check = True
    elif selected_specification == "AT2":
        # always[0,10](RPM < 4750)
        specification = STL.Global(0, 10, FalsifySTL.StrictlyLessThan(1, 0, 0, 4750, S("RPM")))

        strict_horizon_check = True
    elif selected_specification.startswith("AT5"):
        # This is modified from ARCH-COMP to include the next operator which is
        # available as we use discrete time STL.
        # always[0,30]( ( not(GEAR == {0}) and (eventually[0.001,0.1](GEAR == {0})) ) implies ( eventually[0.001,0.1]( always[0,2.5](GEAR == {0}) ) ) )"
        G = int(selected_specification[-1])
        # not(GEAR == {0}) and (eventually[0.001,0.1](GEAR == {0}))
        L = STL.And(STL.Not(STL.Equals(1, 0, 0, G, S("GEAR"))), STL.Next(STL.Equals(1, 0, 0, G, S("GEAR"))))
        # eventually[0.001,0.1]( always[0,2.5](GEAR == {0}) )
        R = STL.Next(STL.Global(0, 2.5, STL.Equals(1, 0, 0, G, S("GEAR"))))

        specification = STL.Global(0, 30, STL.Implication(L, R))

        strict_horizon_check = False
    elif selected_specification.startswith("AT6"):
        A = selected_specification[-1]

        def getSpecification(A):
            if A == "A":
                UB = 4
                SL = 35
            elif A == "B":
                UB = 8
                SL = 50
            else:
                UB = 20
                SL = 65
              
            # (always[0,30](RPM < 3000)) implies (always[0,{0}](SPEED < {1}))
            L = STL.Global(0, 30, FalsifySTL.StrictlyLessThan(1, 0, 0, 3000, S("RPM")))
            R = STL.Global(0, UB, FalsifySTL.StrictlyLessThan(1, 0, 0, SL, S("SPEED")))
            return STL.Implication(L, R)

        if selected_specification.endswith("ABC"):
            specification = STL.And(STL.And(getSpecification("A"), getSpecification("B")), getSpecification("C"))
        else:
            specification = getSpecification(A)

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
        "noise_batch_size": 512,
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

if __name__ == "__main__":
    selected_specification = sys.argv[1]
    N = int(sys.argv[2])
    init_seed = int(sys.argv[3])

    description = "Automatic Transmission"

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

