import unittest

from stgem.budget import Budget
from stgem.generator import STGEM, Search
from stgem.sut.matlab.sut import Matlab_Simulink
from stgem.objective import FalsifySTL
from stgem.objective_selector.objective_selector import ObjectiveSelectorMAB

from stgem.algorithm.ogan.algorithm import OGAN
from stgem.algorithm.ogan.model import OGAN_Model
from stgem.algorithm.random.algorithm import Random
from stgem.algorithm.random.model import Uniform

sut_parameters = {
    "model_file": "../problems/arch-comp/at/Autotrans_shift",
    "inputs": ["THROTTLE", "BRAKE"],
    "outputs": ["SPEED", "RPM", "GEAR"],
    "input_range": [[0, 100], [0, 325]],
    "output_range": [[0, 200], [0, 7000], [0, 4]],
    "simulation_time": 30,
    "time_slices": [5, 5],
    "sampling_step": 0.2
}
specification = "(always[0,30](RPM < 3000)) implies (always[0,4](SPEED < 35))"
mode = "stop_at_first_objective"

class TestPython(unittest.TestCase):
    def test_python(self):
        generator = STGEM(
            description="Matlab-AT/OGAN",
            sut=Matlab_Simulink(sut_parameters),
            budget=Budget(),
            objectives=[FalsifySTL(specification=specification)],
            objective_selector=ObjectiveSelectorMAB(warm_up=60),
            steps=[
                Search(budget_threshold={"executions": 50},
                       mode=mode,
                       algorithm=Random(model_factory=(lambda: Uniform()))),
                Search(budget_threshold={"executions": 300},
                       mode=mode,
                       algorithm=OGAN(model_factory=(lambda: OGAN_Model()))
                )
            ]
        )

        r = generator.run()
        r.dump_to_file("mo3k_python_results.pickle")

if __name__ == "__main__":
    unittest.main()

