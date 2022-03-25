import unittest

import numpy as np
import pandas as pd

from stgem.sut import SUT, SUTResult
from stgem.objective.objective import FalsifySTL

class DummySUT(SUT):
    def __init__(self, odim, outputs):
        super().__init__()
        self.odim = odim
        self.outputs = outputs
        self.output_range = [None for _ in range(self.odim)]
        self.idim = 2
        self.inputs = ["i1", "i2"]
        self.input_range = [None, None]

class TestSTL(unittest.TestCase):

    def get(self, specification, variables, *args, **kwargs):
        sut = DummySUT(len(variables), variables)
        strict_horizon_check = kwargs["strict_horizon_check"] if "strict_horizon_check" in kwargs else True
        objective = FalsifySTL(specification, strict_horizon_check=strict_horizon_check)
        objective.setup(sut)
        return objective(*args)

    def test_stl(self):
        # Test vector-valued output.
        # ---------------------------------------------------------------------
        output = [3, 0.5]
        variables = ["foo", "bar"]
        specification = "always[0,1](foo > 0 and bar > 0)"
        correct_robustness = 0.5

        robustness = self.get(specification, variables, SUTResult(None, output, None, None, None))
        assert robustness == correct_robustness

        output = [3, -0.5]
        variables = ["foo", "bar"]
        specification = "always[0,1](foo > 0 and bar > 0)"
        correct_robustness = 0

        robustness = self.get(specification, variables, SUTResult(None, output, None, None, None))
        assert robustness == correct_robustness

        # Test signal outputs.
        # ---------------------------------------------------------------------
        t = [0.0, 0.5, 1.0, 1.5, 2.0]
        s1 = [4.0, 6.0, 2.0, 8.0, -1.0]
        s2 = [3.0, 6.0, 1.0, 0.5,  3.0]
        signals = [s1, s2]
        variables = ["s1", "s2"]
        specification = "always[0,1](s1 > 0 and s2 > 0)"
        correct_robustness = 1.0

        robustness = self.get(specification, variables, SUTResult(None, signals, t, t, None))
        assert robustness == correct_robustness

        data = pd.read_csv("data/stl_at.csv")
        t = data["time"].tolist()
        s1 = data["SPEED"].tolist()
        s2 = data["RPM"].tolist()
        # The following holds for the signals:
        # always[0,30](RPM < 3000) is true, maximum is 2995.4899293611793.
        # We have SPEED < 39.55047897841963 [0, 4].
        # We have SPEED < 34.221740400656785 in [0, 8].
        # We have SPEED < 45.063039901135426 in [0, 20]:
        signals = [s1, s2]
        variables = ["SPEED", "RPM"]
        specification = "(always[0,30](RPM < 3000)) implies (always[0,4](SPEED < 35))"
        correct_robustness = 0

        robustness = self.get(specification, variables, SUTResult(None, signals, t, t, None))
        assert robustness == correct_robustness

        specification = "(always[0,30](RPM < 3000)) implies (always[0,8](SPEED < 50))"
        correct_robustness = 1

        robustness = self.get(specification, variables, SUTResult(None, signals, t, t, None))
        assert robustness == correct_robustness

        specification = "(always[0,30](RPM < 3000)) implies (always[0,20](SPEED < 65))"
        correct_robustness = 1

        robustness = self.get(specification, variables, SUTResult(None, signals, t, t, None))
        assert robustness == correct_robustness

        # Test time horizon.
        # ---------------------------------------------------------------------
        t = np.linspace(0, 10, 20)
        #     0  0.52 1.05 1.57 2.10 2.63 3.15 3.68 4.21 4.73 5.26 5.78 6.31 6.84 7.36 7.89 8.42 8.94 9.47 10
        s1 = [0, 0,   0,   6,   0,   0,   6,   0,   0,   6,   0,   0,   5,   0,   0,   0,   0,   0,   6,   0]
        s2 = [0, 0,   0,   0,   0,   0,   4,   0,   0,   0,   4,   0,   4,   4,   0,   0,   0,   0,   4,   0]
        variables = ["s1", "s2"]
        signals = [s1, s2]
        specification = "always[0,10]( (s1 > 5) implies (eventually[0,1](s2 < 3)) )"
        correct_robustness = 0

        # Check with strict horizon check.
        try:
            robustness = self.get(specification, variables, SUTResult(None, signals, None, t, None), strict_horizon_check=True)
            assert False
        except Exception:
            pass
        # Check without strict horizon check.
        robustness = self.get(specification, variables, SUTResult(None, signals, None, t, None), strict_horizon_check=False)
        assert robustness == correct_robustness

        # Test time series adjustment.
        # ---------------------------------------------------------------------
        t1 = [0, 1, 2, 3]
        i1 = [1, 3, 4, 1]
        t2 = [0, 0.5, 1, 2, 2.5, 3]
        s1 = [2, 2, 2, 2, 2, 2]
        variables = ["s1"]
        specification = "always[0,3](i1 > s1)"
        correct_robustness = 0

        robustness = self.get(specification, variables, SUTResult([i1], [s1], t1, t2, None))
        assert robustness == correct_robustness

if __name__ == "__main__":
    unittest.main()

