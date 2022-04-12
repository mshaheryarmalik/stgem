from math import pi
import subprocess

import tltk_mtl as STL

from stgem.generator import STGEM, Search
from stgem.budget import Budget
from stgem.sut.matlab import Matlab
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.model import Uniform, LHS
from stgem.objective import FalsifySTL
from stgem.objective_selector import ObjectiveSelectorMAB

mode = "stop_at_first_objective"
specifications = ["F16", # F16
                 ]
selected_specification = "F16"

# Running the model requires Control System Toolbox in Matlab.

# Notice that here the input is a vector.

if selected_specification == "F16":
    # always[0,15] ALTITUDE > 0
    specification = FalsifySTL.StrictlyGreaterThan(1, 0, 0, 0, STL.Signal("ALTITUDE"))

    strict_horizon_check = True
else:
    raise Exception("Unknown specification '{}'.".format(selected_specification))

roll_range = [0.2*pi, 0.2833*pi]
pitch_range = [-0.4*pi, -0.35*pi]
yaw_range = [-0.375*pi, -0.125*pi]
sut_parameters = {"model_file": "problems/arch-comp-2021/f16/run_f16",
                  "init_model_file": "problems/arch-comp-2021/f16/init_f16",
                  "input_type": "vector",
                  "output_type": "signal",
                  "inputs": ["ROLL", "PITCH", "YAW"],
                  "outputs": ["ALTITUDE"],
                  "input_range": [roll_range, pitch_range, yaw_range],
                  "output_range": [[0, 4040]], # Starting altitude defined in init_f16.m.
                  "simulation_time": 15,
                 }

ogan_parameters = {"fitness_coef": 0.95,
                   "train_delay": 1,
                   "N_candidate_tests": 1
                   }

ogan_model_parameters = {"optimizer": "Adam",
                         "discriminator_lr": 0.005,
                         "discriminator_betas": [0.9, 0.999],
                         "generator_lr": 0.0010,
                         "generator_betas": [0.9, 0.999],
                         "noise_batch_size": 512,
                         "generator_loss": "MSE",
                         "discriminator_loss": "MSE",
                         "generator_mlm": "GeneratorNetwork",
                         "generator_mlm_parameters": {"noise_dim": 20, "neurons": 64},
                         "discriminator_mlm": "DiscriminatorNetwork",
                         "discriminator_mlm_parameters": {"neurons": 64, "discriminator_output_activation": "sigmoid"},
                         "train_settings_init": {"epochs": 2, "discriminator_epochs": 20, "generator_batch_size": 32},
                         "train_settings": {"epochs": 1, "discriminator_epochs": 30, "generator_batch_size": 32}
                        }

generator = STGEM(
                  description="Airfract Ground Collision Avoidance System",
                  sut=Matlab(sut_parameters),
                  budget=Budget(),
                  objectives=[FalsifySTL(specification=specification, strict_horizon_check=strict_horizon_check)],
                  objective_selector=ObjectiveSelectorMAB(warm_up=20),
                  steps=[
                         Search(mode=mode,
                                budget_threshold={"executions": 20},
                                algorithm=Random(model_factory=(lambda: Uniform()))),
                         Search(mode=mode,
                                budget_threshold={"executions": 40},
                                algorithm=OGAN(model_factory=(lambda: OGAN_Model(ogan_model_parameters)), parameters=ogan_parameters))
                        ]
                  )

if __name__ == "__main__":
    r = generator.run()

