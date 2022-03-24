import math

import unittest

from stgem.generator import STGEM, Search
from stgem.sut.python.sut import PythonFunction
from stgem.objective import Minimize
from stgem.objective_selector import ObjectiveSelectorMAB
from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform

def myfunction(input: [[-15, 15], [-15, 15], [-15, 15]]) -> [[0, 350], [0, 350], [0, 350]]:
    x1, x2, x3 = input[0], input[1], input[2]
    h1 = 305 - 100 * (math.sin(x1 / 3) + math.sin(x2 / 3) + math.sin(x3 / 3))
    h2 = 230 - 75 * (math.cos(x1 / 2.5 + 15) + math.cos(x2 / 2.5 + 15) + math.cos(x3 / 2.5 + 15))
    h3 = (x1 - 7) ** 2 + (x2 - 7) ** 2 + (x3 - 7) ** 2 - (
            math.cos((x1 - 7) / 2.75) + math.cos((x2 - 7) / 2.75) + math.cos((x3 - 7) / 2.75))

    return [h1, h2, h3]

class TestPython(unittest.TestCase):
    def test_ogan(self):
        mode = "stop_at_first_objective"

        ogan_model_parameters = {"optimizer": "Adam",
                                 "discriminator_lr": 0.005,
                                 "discriminator_betas": [0.9, 0.999],
                                 "generator_lr": 0.0010,
                                 "generator_betas": [0.9, 0.999],
                                 "noise_batch_size": 512,
                                 "generator_loss": "MSE",
                                 "discriminator_loss": "MSE",
                                 "generator_mlm": "GeneratorNetwork",
                                 "generator_mlm_parameters": {"noise_dim": 20, "output_shape": 3, "neurons": 64},
                                 "discriminator_mlm": "DiscriminatorNetwork",
                                 "discriminator_mlm_parameters": {"input_shape": 3, "neurons": 64, "discriminator_output_activation": "sigmoid"},
                                 "train_settings_init": {"epochs": 2, "discriminator_epochs": 20, "generator_batch_size": 32},
                                 "train_settings": {"epochs": 1, "discriminator_epochs": 30, "generator_batch_size": 32}
                                }

        generator = STGEM(
            description="mo3d/OGAN",
            sut=PythonFunction(function=myfunction),
            objectives=[Minimize(selected=[0], scale=True),
                        Minimize(selected=[1], scale=True),
                        Minimize(selected=[2], scale=True)
                        ],
            objective_selector=ObjectiveSelectorMAB(warm_up=5),
            steps=[
                Search(max_tests=20,
                       mode=mode,
                       algorithm=Random(model_factory=(lambda: Uniform()))),
                Search(max_tests=5,
                       mode=mode,
                       algorithm=OGAN(model_factory=(lambda: OGAN_Model(ogan_model_parameters))))
            ]
        )

        r = generator.run()
        r.dump_to_file("mo3k_python_wogan_results.pickle")

if __name__ == "__main__":
    unittest.main()

