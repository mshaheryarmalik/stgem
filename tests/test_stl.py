import unittest, traceback

import numpy as np
import pandas as pd

import tltk_mtl as STL

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
        scale = kwargs["scale"] if "scale" in kwargs else True
        clip = kwargs["clip"] if "clip" in kwargs else True
        strict_horizon_check = kwargs["strict_horizon_check"] if "strict_horizon_check" in kwargs else True
        objective = FalsifySTL(specification, scale=scale, clip=clip, strict_horizon_check=strict_horizon_check)
        objective.setup(sut)
        return objective(*args), objective.horizon

    def test_stl(self):
        # Test vector-valued output.
        # ---------------------------------------------------------------------
        output = [3, 0.5]
        variables = ["foo", "bar"]
        # foo > 0 and bar > 0
        L = STL.Signal("foo")
        R = STL.Signal("bar")
        specification = STL.And(L, R)
        correct_robustness = 0.5

        robustness, _ = self.get(specification, variables, SUTResult(None, output, None, None, None))
        assert robustness == correct_robustness

        output = [3, -0.5]
        variables = ["foo", "bar"]
        # always[0,1](foo > 0 and bar > 0)
        L = STL.Signal("foo")
        R = STL.Signal("bar")
        specification = STL.Global(0, 1, STL.And(L, R))
        correct_robustness = 0

        robustness, _ = self.get(specification, variables, SUTResult(None, output, None, None, None))
        assert robustness == correct_robustness

        # Test signal outputs.
        # ---------------------------------------------------------------------
        t = [0.0, 0.5, 1.0, 1.5, 2.0]
        s1 = [4.0, 6.0, 2.0, 8.0, -1.0]
        s2 = [3.0, 6.0, 1.0, 0.5,  3.0]
        signals = [s1, s2]
        variables = ["s1", "s2"]
        # always[0,1](s1 >= 0 and s2 >= 0)
        L = FalsifySTL.GreaterThan(1, 0, 0, 0, STL.Signal("s1"))
        R = FalsifySTL.GreaterThan(1, 0, 0, 0, STL.Signal("s2"))
        specification = STL.And(L, R)
        specification = STL.Global(0, 1, specification)
        sp = 0.5
        correct_robustness = 1.0

        robustness, horizon = self.get(specification, variables, SUTResult(None, signals, None, t, None), sampling_period=sp)
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
        scale = False
        # always[0,30](RPM <= 3000)) implies (always[0,4](SPEED <= 35)
        L = STL.Global(0, 30, STL.Predicate("RPM", 1, 3000))
        R = STL.Global(0, 4, STL.Predicate("SPEED", 1, 35))
        specification = STL.Implication(L, R)
        sp = 0.01
        correct_robustness = 0

        robustness, horizon = self.get(specification, variables, SUTResult(None, signals, None, t, None), scale=scale, sampling_period=sp)
        assert horizon == 30
        assert robustness == correct_robustness

        # always[0,30](RPM < 3000)) implies (always[0,8](SPEED < 50)
        R = STL.Global(0, 8, STL.Predicate("SPEED", 1, 50))
        specification = STL.Implication(L, R)
        correct_robustness = 1

        robustness, _ = self.get(specification, variables, SUTResult(None, signals, None, t, None), scale=scale, sampling_period=sp)
        assert robustness == correct_robustness

        # always[0,30](RPM < 3000)) implies (always[0,20](SPEED < 65)
        R = STL.Global(0, 20, STL.Predicate("SPEED", 1, 65))
        specification = STL.Implication(L, R)
        correct_robustness = 1

        robustness, _ = self.get(specification, variables, SUTResult(None, signals, None, t, None), scale=scale, sampling_period=sp)
        assert robustness == correct_robustness

        # Test time horizon.
        # ---------------------------------------------------------------------
        t = [0.5*i for i in range(21)]
        #     0  0.5 1.0 1.5 2.0 2.5 3.0 3.5 4.0 4.5 5.0 5.5 6.0 6.5 7.0 7.5 8.0 8.5 9.0 9.5 10
        s1 = [0, 0,  0,  6,  0,  0,  6,  0,  0,  6,  0,  0,  5,  0,  0,  0,  0,  0,  6,  0,  0]
        s2 = [0, 0,  0,  0,  0,  0,  4,  0,  0,  0,  4,  0,  4,  4,  4,  0,  0,  0,  4,  0,  0]
        variables = ["s1", "s2"]
        signals = [s1, s2]
        # always[0,10]( (s1 >= 5) implies (eventually[0,1](s2 <= 3)) )
        L = STL.Predicate("s1", -1, 5)
        R = STL.Finally(0, 1, STL.Predicate("s2", 1, 3))
        specification = STL.Global(0, 10, STL.Implication(L, R))
        correct_robustness = 0

        # Check with strict horizon check.
        try:
            robustness, _ = self.get(specification, variables, SUTResult(None, signals, None, t, None), scale=scale, strict_horizon_check=True)
        except Exception as E:
            if not E.args[0].startswith("The horizon"):
                traceback.print_exc()
                raise
        # Check without strict horizon check.
        robustness, horizon = self.get(specification, variables, SUTResult(None, signals, None, t, None), scale=scale, strict_horizon_check=False)
        assert horizon == 11
        assert robustness == correct_robustness

        # Test time series adjustment.
        # ---------------------------------------------------------------------
        t1 = [0, 1, 2, 3]
        i1 = [1, 3, 4, 1]
        t2 = [0, 0.5, 1, 2, 2.5, 3]
        s1 = [2, 2, 2, 2, 2, 2]
        variables = ["s1"]
        # always[0,3](i1 >= s1)
        L = STL.Signal("i1")
        R = STL.Signal("s1")
        specification = FalsifySTL.GreaterThan(1, 0, 1, 0, L, R)
        specification = STL.Global(0, 3, specification)
        sp = 1
        correct_robustness = 0

        robustness, horizon = self.get(specification, variables, SUTResult([i1], [s1], t1, t2, None), scale=scale, sampling_period=sp)
        assert horizon == 3
        assert robustness == correct_robustness

if __name__ == "__main__":
    unittest.main()

