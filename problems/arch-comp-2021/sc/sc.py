import tltk_mtl as STL

from stgem.generator import STGEM, Search
from stgem.budget import Budget
from stgem.sut.matlab.sut import Matlab
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.model import Uniform, LHS
from stgem.objective import FalsifySTL
from stgem.objective_selector import ObjectiveSelectorMAB

mode = "stop_at_first_objective"
scale = False
specifications = ["SC", # SC
                 ]
selected_specification = "SC"

# Running the model requires Deep Learning Toolbox in Matlab.

# Notice that this only implements the Instance 2 version of the problem where
# the input signal is split into exactly 20 segments.

sut_parameters = {"model_file": "problems/arch-comp-2021/sc/run_steamcondenser",
                  "input_type": "piecewise constant signal",
                  "output_type": "signal",
                  "outputs": ["PRESSURE"],
                  "input_range": [[3.99, 4.01]],
                  "simulation_time": 35,
                  "time_slices": [1.75],
                  "sampling_step": 0.5
                 }

asut = Matlab(sut_parameters)

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
                  description="Steam Condenser with Recurrent Neural Network Controller",
                  sut=asut,
                  budget=Budget(),
                  objectives=[FalsifySTL(specification=specification, scale=scale, strict_horizon_check=strict_horizon_check)],
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

