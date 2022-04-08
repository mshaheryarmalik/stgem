import tltk_mtl as STL

from stgem.generator import STGEM, Search
from stgem.budget import Budget
from stgem.sut.matlab.sut import Matlab_Simulink
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.model import Uniform, LHS
from stgem.objective import FalsifySTL
from stgem.objective_selector import ObjectiveSelectorMAB

mode = "stop_at_first_objective"
selected_specification = "AT6ABC"

if selected_specification == "AT1":
    # always[0,20](SPEED < 120)
    specification = STL.Global(0, 20, STL.LessThan(1, 0, 0, 120, STL.Signal("SPEED")))

    strict_horizon_check = True
elif selected_specification == "AT2":
    # always[0,10](RPM < 4750)
    specification = STL.Global(0, 10, STL.LessThan(1, 0, 0, 4750, STL.Signal("RPM")))
elif selected_specification.startswith("AT5"):
    # This is modified from ARCH-COMP to include the next operator which is
    # available as we use discrete time STL.
    # always[0,30]( ( not(GEAR == {0}) and (eventually[0.001,0.1](GEAR == {0})) ) implies ( eventually[0.001,0.1]( always[0,2.5](GEAR == {0}) ) ) )"
    G = int(selected_specification[-1])
    # not(GEAR == {0}) and (eventually[0.001,0.1](GEAR == {0}))
    L = STL.And(STL.Not(STL.Equals(1, 0, 0, G, STL.Signal("GEAR"))), STL.Next(STL.Equals(1, 0, 0, G, STL.Signal("GEAR"))))
    # eventually[0.001,0.1]( always[0,2.5](GEAR == {0}) )
    R = STL.Next(STL.Global(0, 2.5, STL.Equals(1, 0, 0, G, STL.Signal("GEAR"))))

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
        L = STL.Global(0, 30, STL.LessThan(1, 0, 0, 3000, STL.Signal("RPM")))
        R = STL.Global(0, UB, STL.LessThan(1, 0, 0, SL, STL.Signal("SPEED")))
        return STL.Implication(L, R)

    if selected_specification.endswith("ABC"):
        specification = STL.And(STL.And(getSpecification("A"), getSpecification("B")), getSpecification("C"))
    else:
        specification = getSpecification(A)

    strict_horizon_check = True
else:
    raise Exception("Unknown specification '{}'.".format(selected_specification))

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
        "generator_loss": "MSE",
        "discriminator_loss": "MSE",
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

generator = STGEM(
                  description="Automatic Transmission",
                  sut=Matlab_Simulink(sut_parameters),
                  budget=Budget(),
                  objectives=[FalsifySTL(specification=specification, strict_horizon_check=strict_horizon_check)],
                  objective_selector=ObjectiveSelectorMAB(warm_up=20),
                  steps=[
                         Search(mode=mode,
                                budget_threshold={"executions": 20},
                                algorithm=Random(model_factory=(lambda: Uniform()))),
                         Search(mode=mode,
                                budget_threshold={"executions": 40},
                                algorithm=OGAN(model_factory=(lambda: OGAN_Model(ogan_model_parameters["convolution"])), parameters=ogan_parameters))
                        ]
                  )

if __name__ == "__main__":
    r = generator.run()

