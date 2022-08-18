import unittest, traceback

import numpy as np
import pandas as pd

from stgem.sut import SUT, SUTInput, SUTOutput
from stgem.objective.objective import FalsifySTL
import stl.robustness as STL

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

    def get(self, specification, variables, sut_input, sut_output, ranges=None, scale=False, strict_horizon_check=True):
        sut = DummySUT(len(variables), variables)
        objective = FalsifySTL(specification, ranges=ranges, scale=scale, strict_horizon_check=strict_horizon_check)
        objective.setup(sut)
        return objective(sut_input, sut_output), objective

    def test_stl(self):
        # Test vector-valued output.
        # ---------------------------------------------------------------------
        output = [3, 0.5]
        variables = ["foo", "bar"]
        specification = "foo > 0 and bar > 0"
        correct_robustness = 0.5

        robustness, _ = self.get(specification, variables, SUTInput(None, None, None), SUTOutput(output, None, None))
        assert robustness == correct_robustness

        output = [3, -0.5]
        variables = ["foo", "bar"]
        specification = "always[0,1] (foo > 0 and bar > 0)"
        correct_robustness = -0.5

        robustness, _ = self.get(specification, variables, SUTInput(None, None, None), SUTOutput(output, None, None))
        assert robustness == correct_robustness

        # Test signal outputs.
        # ---------------------------------------------------------------------
        t = [0.0, 0.5, 1.0, 1.5, 2.0]
        s1 = [4.0, 6.0, 2.0, 8.0, -1.0]
        s2 = [3.0, 6.0, 1.0, 0.5,  3.0]
        signals = [s1, s2]
        variables = ["s1", "s2"]
        specification = "always[0,1] (s1 >= 0 and s2 >= 0)"
        correct_robustness = 1.0

        robustness, _ = self.get(specification, variables, SUTInput(None, None, None), SUTOutput(signals, t, None))
        assert robustness == correct_robustness

        data = pd.read_csv("data/stl_at.csv")
        t = data["time"].tolist()
        s1 = data["SPEED"].tolist()
        s2 = data["RPM"].tolist()
        # The following holds for the signals:
        # always[0,30](RPM < 3000) is true, maximum is 2995.4899293611793.
        # We have SPEED < 39.55047897841963 [0, 4].
        # We have SPEED < 45.06303990113543 in [0, 8].
        # We have SPEED < 45.063039901135426 in [0, 20]:
        signals = [s1, s2]
        variables = ["SPEED", "RPM"]
        scale = False
        specification = "(always[0,30] RPM <= 3000) -> (always[0,4] SPEED <= 35)"
        correct_robustness = -4.55048

        robustness, objective = self.get(specification, variables, SUTInput(None, None, None), SUTOutput(signals, t, None), scale=scale)
        assert abs(robustness - correct_robustness) < 1e-5

        specification = "(always[0,30] RPM <= 3000) -> (always[0,8] SPEED < 50)"
        correct_robustness = 4.936960098864567

        robustness, _ = self.get(specification, variables, SUTInput(None, None, None), SUTOutput(signals, t, None), scale=scale)
        assert abs(robustness - correct_robustness) < 1e-5

        specification = "(always[0,30] RPM <= 3000) -> (always[0,20] SPEED < 65)"
        correct_robustness = 19.936958

        robustness, _ = self.get(specification, variables, SUTInput(None, None, None), SUTOutput(signals, t, None), scale=scale)
        assert abs(robustness == correct_robustness) < 1e-5

        # Test time horizon.
        # ---------------------------------------------------------------------
        t = [0.5*i for i in range(21)]
        #     0  0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10
        s1 = [0, 0,  0,  6,  0,  0,  6,  0,  0,  6,  0,  0,  5,  0,  0,  0,  0,  0,  6,  0,  0]
        s2 = [0, 0,  0,  0,  0,  0,  4,  0,  0,  0,  4,  0,  4,  4,  4,  0,  0,  0,  4,  0,  0]
        variables = ["s1", "s2"]
        signals = [s1, s2]
        specification = "always[0,10] ( (s1 >= 5) -> (eventually[0,1] s2 <= 3) )"
        correct_robustness = 0

        # Check with strict horizon check.
        try:
            robustness, _ = self.get(specification, variables, SUTInput(None, None, None), SUTOutput(signals, t, None), scale=scale, strict_horizon_check=True)
        except Exception as E:
            if not E.args[0].startswith("The horizon"):
                traceback.print_exc()
                raise
        # Check without strict horizon check.
        robustness, objective = self.get(specification, variables, SUTInput(None, None, None), SUTOutput(signals, t, None), scale=scale, strict_horizon_check=False)
        assert objective.horizon == 11
        assert robustness == correct_robustness

        # Test time series adjustment.
        # ---------------------------------------------------------------------
        t1 = [0, 1, 2, 3]
        i1 = [1, 3, 4, 1]
        t2 = [0, 0.5, 1, 2, 2.5, 3]
        s1 = [2, 2, 2, 2, 2, 2]
        variables = ["s1"]
        specification = "always[0,3] i1 >= s1"
        correct_robustness = -1.0

        robustness, objective = self.get(specification, variables, SUTInput(None, [i1], t1), SUTOutput([s1], t2, None), scale=scale)
        assert objective.horizon == 3
        assert robustness == correct_robustness

        # Test signal ranges.
        # ---------------------------------------------------------------------
        t = [0, 1, 2, 3, 4, 5]
        s1 = [100, 150, 70, 30, 190, 110]   # scale [0, 200]
        s2 = [4500, 100, 0, 2300, -100, -5] # scale [-200, 4500]
        variables = ["s1", "s2"]
        signals = [s1, s2]
        specification = "3*s1 <= s2"
        ranges = {"s1": [0, 200], "s2": [-200, 4500]}

        robustness, objective = self.get(specification, variables, SUTInput(None, None, None), SUTOutput(signals, t, None), ranges=ranges, scale=scale)
        assert objective.specification.range == [-800, 4500]

if __name__ == "__main__":
    unittest.main()

