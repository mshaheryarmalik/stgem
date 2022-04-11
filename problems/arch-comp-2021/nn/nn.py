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
specifications = ["NN", # NN
                  "NNX" # NNX
                 ]
selected_specification = "NN"

# Notice that this only implements the Instance 2 version of the problem where
# the input signal is split into exactly 3 segments.

# Some ARCH-COMP specifications have requirements whose horizon is longer than
# the output signal for some reason. Thus strict horizon check needs to be
# disabled in some cases.
if selected_specification == "NN":
    alpha = 0.005
    beta = 0.03
    # inequality := |POS - REF| > alpha + beta*|REF|
    # We make two copies in order to not share state.
    inequality1 = STL.Not(STL.LessThan(1, 0, beta, alpha, STL.Abs(STL.Subtract(STL.Signal("POS"), STL.Signal("REF"))), STL.Abs(STL.Signal("REF"))))
    inequality2 = STL.Not(STL.LessThan(1, 0, beta, alpha, STL.Abs(STL.Subtract(STL.Signal("POS"), STL.Signal("REF"))), STL.Abs(STL.Signal("REF"))))
    # always[1,37]( inequality implies (always[0,2]( eventually[0,1] not inequality )) )
    specification = STL.Global(1, 37, STL.Implication(inequality1, STL.Global(0, 2, STL.Finally(0, 1, STL.Not(inequality2)))))

    strict_horizon_check = True
elif selected_specification == "NNX":
    # eventually[0,1](POS > 3.2)
    F1 = STL.Finally(0, 1, STL.Not(STL.LessThan(1, 0, 0, 3.2, STL.Signal("POS"))))
    # eventually[1,1.5]( always[0,0.5](1.75 < POS < 2.25) )
    L = STL.Not(STL.LessThan(1, 0, 0, 1.75, STL.Signal("POS")))
    R = STL.Not(STL.LessThan(0, 2.25, 1, 0, None, STL.Signal("POS")))
    inequality = STL.And(L, R)
    F2 = STL.Finally(1, 1.5, STL.Global(0, 0.5, inequality))
    # always[2,3](1.825 < POS < 2.175)
    L = STL.Not(STL.LessThan(1, 0, 0, 1.825, STL.Signal("POS")))
    R = STL.Not(STL.LessThan(0, 2.175, 1, 0, None, STL.Signal("POS")))
    inequality = STL.And(L, R)
    F3 = STL.Global(2, 3, inequality)

    specification = STL.And(F1, STL.And(F2, F3))

    strict_horizon_check = True
else:
    raise Exception("Unknown specification '{}'.".format(selected_specification))

sut_parameters = {"model_file": "problems/arch-comp-2021/nn/run_neural",
                  "init_model_file": "problems/arch-comp-2021/nn/init_neural",
                  "input_type": "piecewise constant signal",
                  "output_type": "signal",
                  "inputs": ["REF"],
                  "outputs": ["POS"],
                  "input_range": [[1, 3]],
                  "simulation_time": 40,
                  "time_slices": [13],
                  "sampling_step": 0.5
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
                  description="Neural-network Controller",
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

