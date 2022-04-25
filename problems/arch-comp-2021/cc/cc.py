import os

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
scale = False
specifications = ["CC1",
                  "CC2"
                  "CC3"
                  "CC4"
                  "CC5"
                  "CCX"
                 ]
selected_specification = "CC1"

def build_specification(selected_specification, asut=None):
    sut_parameters = {"model_file": "problems/arch-comp-2021/cc/cars",
                      "input_type": "piecewise constant signal",
                      "output_type": "signal",
                      "inputs": ["THROTTLE", "BRAKE"],
                      "outputs": ["Y1", "Y2", "Y3", "Y4", "Y5"],
                      "input_range": [[0, 100], [0, 100]],
                      "simulation_time": 100,
                      "time_slices": [5, 5],
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
    if selected_specification == "CC1":
        # always[0,100]( y5 - y4 <= 40 )
        specification = STL.Global(0, 100, STL.LessThan(1, 0, 1, 40, S("Y5"), S("Y4")))
        
        strict_horizon_check = True
    elif selected_specification == "CC2":
        # always[0,70]( eventually[0,30]( y5 - y4 >= 15 ) )
        specification = STL.Global(0, 70, STL.Finally(0, 30, FalsifySTL.GreaterThan(1, 0, 1, 15, S("Y5"), S("Y4"))))

        strict_horizon_check = True
    elif selected_specification == "CC3":
        # always[0,80]( (always[0,20]( y2 - y1 <= 20 )) or (eventually[0,20]( y5 - y4 >= 40 )) )
        L = STL.Global(0, 20, STL.LessThan(1, 0, 1, 20, S("Y2"), S("Y1"))) 
        R = STL.Finally(0, 20, FalsifySTL.GreaterThan(1, 0, 1, 40, S("Y5"), S("Y4")))
        specification = STL.Global(0, 80, STL.And(L, R))

        strict_horizon_check = True
    elif selected_specification == "CC4":
        # always[0,65]( eventually[0,30]( always[0,20]( y5 - y4 >= 8 ) ) )
        specification = STL.Global(0, 65, STL.Finally(0, 30, STL.Global(0, 20, FalsifySTL.GreaterThan(1, 0, 1, 8, S("Y5"), S("Y4")))))

        strict_horizon_check = False
    elif selected_specification == "CC5":
        # always[0,72]( eventually[0,8]( always[0,5]( y2 - y1 >= 9 ) implies always[5,20]( y5 - y4 >= 9 ) ) )
        L = STL.Global(0, 5, FalsifySTL.GreaterThan(1, 0, 1, 9, S("Y2"), S("Y3")))
        R = STL.Global(5, 20, FalsifySTL.GreaterThan(1, 0, 1, 9, S("Y5"), S("Y4")))
        specification = STL.Global(0, 72, STL.Finally(0, 8, STL.Implication(L, R)))

        strict_horizon_check = True
    elif selected_specification == "CCX":
        # always[0,50]( y2 - y1 > 7.5 ) and always[0,50]( y3 - y2 > 7.5 ) and always[0,50]( y4 - y3 > 7.5 ) and always[0,50]( y5 - y4 > 7.5 )
        def getSpecification(N):
            return STL.Global(0, 50, FalsifySTL.StrictlyGreaterThan(1, 0, 1, 7.5, S("Y{}".format(N+1)), S("Y{}".format(N))))

        F1 = getSpecification(1)
        F2 = getSpecification(2)
        F3 = getSpecification(3)
        F4 = getSpecification(4)
        specification = STL.And(F1, STL.And(F2, STL.And(F3, F4) ) )

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
    N = 1

    description = "Chasing Cars"

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
            c = 25321
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

