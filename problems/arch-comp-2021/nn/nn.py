import os, sys

from stgem.generator import STGEM, Search, run_multiple_generators
from stgem.budget import Budget
from stgem.objective import FalsifySTL
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.model import Uniform, LHS
from stgem.objective_selector import ObjectiveSelectorAll, ObjectiveSelectorMAB

from util import build_specification

mode = "stop_at_first_objective"

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

if __name__ == "__main__":
    selected_specification = sys.argv[1]
    N = int(sys.argv[2])
    init_seed = int(sys.argv[3])
    identifier = sys.argv[4] if len(sys.argv) >= 5 else None

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
        asut, specifications, scale, strict_horizon_check = build_specification(selected_specification, asut)
        budget = Budget()
        epsilon = 0.01
        objectives = [FalsifySTL(specification=specification, epsilon=epsilon, scale=scale, strict_horizon_check=strict_horizon_check) for specification in specifications]
        objective_selector = ObjectiveSelectorMAB(warm_up=100)
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
        file_name = "{}{}_{}.pickle".format(selected_specification, "_" + identifier if identifier is not None else "", str(result.timestamp).replace(" ", "_"))
        result.dump_to_file(os.path.join("output", file_name))

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

