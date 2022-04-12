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

afc_mode = "normal" # normal/power
if afc_mode == "normal":
    throttle_range = [0, 61.2]
elif afc_mode == "power":
    throttle_range = [61.2, 81.2]

mode = "stop_at_first_objective"
specifications = ["AFC27", # AFC27, normal
                  "AFC29"  # AFC29,AFC33 normal/power
                 ]
selected_specification = "AFC27"

# Some ARCH-COMP specifications have requirements whose horizon is longer than
# the output signal for some reason. Thus strict horizon check needs to be
# disabled in some cases.
S = lambda var: STL.Signal(var)
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

sut_parameters = {"model_file": "problems/arch-comp-2021/afc/run_powertrain",
                  "init_model_file": "problems/arch-comp-2021/afc/init_powertrain",
                  "input_type": "piecewise constant signal",
                  "output_type": "signal",
                  "inputs": ["THROTTLE", "ENGINE"],
                  "outputs": ["MU", "MODE"],
                  "input_range": [throttle_range, [900, 1100]],
                  "simulation_time": 50,
                  "time_slices": [10, 50],
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
                  description="Fuel Control of an Automotive Powertrain ({} mode)".format(afc_mode),
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

