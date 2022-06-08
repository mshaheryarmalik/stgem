import numpy as np

from stgem.generator import STGEM, Search
from stgem.algorithm import Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform, LHS
from stgem.algorithm.wogan.algorithm import WOGAN
from stgem.algorithm.wogan.model import WOGAN_Model
from stgem.objective import Objective
from stgem.objective_selector import ObjectiveSelectorAll

from sut import SBSTSUT, SBSTSUT_validator

class UniformDependent(Model):
    """Model for uniformly random search which does not select components
    independently."""

    def generate_test(self):
        # The components of the actual test are curvature values in the input
        # range (default [-0.07, 0.07]). Here we do not choose the components
        # of a test independently in [-1, 1] but we do as in the Frenetic
        # algorithm where the next component is in the range of the previous
        # value +- 0.05 (in the scale [-0.07, 0.07]).

        test = np.zeros(self.search_space.input_dimension)
        test[0] = np.random.uniform(-1, 1)
        for i in range(1, len(test)):
            test[i] = max(-1, min(1, test[i - 1] + (0.05/0.07) * np.random.uniform(-1, 1)))

        return test

class MaxOOB(Objective):
    """Objective which picks the maximum M from the first output signal and
    returns 1-M for minimization."""

    def __call__(self, t, r):
        #return 1 - max(r.outputs)
        return 1 - max(r.outputs[0])

mode = "stop_at_first_objective"

sut_parameters = {
    "beamng_home":  "C:/BeamNG/BeamNG.tech.v0.24.0.1",
    "curvature_points": 5,
    "curvature_range": 0.07,
    "step_size": 15,
    "map_size": 200,
    "max_speed": 75.0
}

wogan_parameters = {
    "bins": 10,
    "wgan_batch_size": 32,
    "fitness_coef": 0.95,
    "train_delay": 3,
    "N_candidate_tests": 1,
    "shift_function": "linear",
    "shift_function_parameters": {"initial": 0, "final": 3},
}

wogan_model_parameters = {
    "critic_optimizer": "Adam",
    "critic_lr": 0.001,
    "critic_betas": [0, 0.9],
    "generator_optimizer": "Adam",
    "generator_lr": 0.001,
    "generator_betas": [0, 0.9],
    "noise_batch_size": 32,
    "gp_coefficient": 10,
    "eps": 1e-6,
    "report_wd": True,
    "analyzer": "Analyzer_NN",
    "analyzer_parameters": {
        "optimizer": "Adam",
        "lr": 0.005,
        "betas": [0, 0.9],
        "loss": "MSE,logit",
        "l2_regularization_coef": 0.001,
        "analyzer_mlm": "AnalyzerNetwork",
        "analyzer_mlm_parameters": {
            "hidden_neurons": [64, 64],
            "layer_normalization": False
        },
    },
    "generator_mlm": "GeneratorNetwork",
    "generator_mlm_parameters": {
        "noise_dim": 20,
        "hidden_neurons": [128, 128],
        "batch_normalization": False,
        "layer_normalization": False
    },
    "critic_mlm": "CriticNetwork",
    "critic_mlm_parameters": {
        "hidden_neurons": [128, 128]
    },
    "train_settings_init": {
        "epochs": 3,
        "analyzer_epochs": 20,
        "critic_steps": 5,
        "generator_steps": 1
    },
    "train_settings": {
        "epochs": 10,
        "analyzer_epochs": 1,
        "critic_steps": 5,
        "generator_steps": 1
    },
}

generator = STGEM(
                  description="SBST 2022 BeamNG.tech simulator",
                  sut=SBSTSUT(sut_parameters),
                  #sut=SBSTSUT_validator(sut_parameters),
                  objectives=[MaxOOB()],
                  objective_selector=ObjectiveSelectorAll(),
                  steps=[
                         Search(mode=mode,
                                budget_threshold={"executions": 50},
                                algorithm=Random(model_factory=(lambda: UniformDependent()))),
                         Search(mode=mode,
                                budget_threshold={"executions": 200},
                                algorithm=WOGAN(model_factory=(lambda: WOGAN_Model(wogan_model_parameters)), parameters=wogan_parameters))
                        ]
                  )

if __name__ == "__main__":
    r = generator.run()
