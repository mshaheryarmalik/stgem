import os, sys

from stgem.generator import Search
from stgem.experiment import Experiment
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.ogan.model_keras import OGANK_Model
from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model
from stgem.algorithm.random.model import Uniform, LHS
from stgem.objective_selector import ObjectiveSelectorAll

from util import build_specification, get_sut_objective_factory

sys.path.append(os.path.join("problems", "arch-comp-2021"))
from common import get_generator_factory, get_seed_factory

# Running the model requires Control System Toolbox in Matlab.

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

def objective_selector_factory():
    return ObjectiveSelectorAll()

def step_factory():
    step_1 = Search(mode=mode,
                    budget_threshold={"executions": 75},
                    #algorithm=Random(model_factory=(lambda: LHS(parameters={"samples": 75})))
                    algorithm=Random(model_factory=(lambda: Uniform()))
                   )      
    step_2 = Search(mode=mode,
                    budget_threshold={"executions": 300},
                    #algorithm=WOGAN(model_factory=(lambda: WOGAN_Model()))
                    #algorithm=OGAN(model_factory=(lambda: OGANK_Model()))
                    algorithm=OGAN(model_factory=(lambda: OGAN_Model(ogan_model_parameters["dense"])), parameters=ogan_parameters)
                   )
    #steps = [step_1]
    steps = [step_1, step_2]
    return steps

if __name__ == "__main__":
    selected_specification = sys.argv[1]
    N = int(sys.argv[2])
    init_seed = int(sys.argv[3])
    identifier = sys.argv[4] if len(sys.argv) >= 5 else None

    description = "Aircraft Ground Collision Avoidance System"

    def callback(result):
        file_name = "{}{}_{}.pickle".format(selected_specification, "_" + identifier if identifier is not None else "", str(result.timestamp).replace(" ", "_"))
        result.dump_to_file(os.path.join("output", file_name))

    epsilon = 0.0
    sut_factory, objective_factory = get_sut_objective_factory(selected_specification, epsilon)
    experiment = Experiment(N, get_generator_factory(description, sut_factory, objective_factory, objective_selector_factory, step_factory), get_seed_factory(init_seed), result_callback=callback)

    N_workers = 1
    experiment.run(N_workers=N_workers, silent=False)

