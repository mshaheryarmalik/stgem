import numpy as np
from unittest import TestCase
import stgem.objective.Robustness as rbst

class ParserTestCase(TestCase):
    @classmethod
    def setUpClass(cls):
        cls._signals = {
            "signal1": np.array([-1, 1]),
            "signal2": np.array([4, 8]),
        }
        cls._timestamps = np.arange(len(list(cls._signals.values())[0]))
        cls._interval = '({}, {})'.format(cls._timestamps[0], cls._timestamps[-1])  # Interval of timestamp length
        cls._traces = rbst.Traces(cls._timestamps, cls._signals)

        """cls._preds = {
            "pred1": rbst.Signal("pred1", [1, 2]),
            "pred2": rbst.Signal("pred2", [1, 4]),
            "pred3": rbst.Signal("pred3", [1, 8]),
            "pred4": rbst.Signal("pred4", [1, 1]),
            "pred5": rbst.Signal("pred5", [1, -1]),
            "pred6": rbst.Signal("pred6", [-1, 1]),
            "pred7": rbst.Signal("pred7", [-1, -1]),
            "pred8": rbst.Signal("pred8", [1, 110000]),
            "pred9": rbst.Signal("pred9", [1, 0.0000112]),
        }"""

        """        cls._preds = {
            "pred1": rbst.Predicate("pred1", np.array([1]), np.array([2])),
            "pred2": rbst.Predicate("pred2", np.array([1]), np.array([4])),
            "pred3": rbst.Predicate("pred3", np.array([1]), np.array([8])),
            "pred4": rbst.Predicate("pred4", np.array([1]), np.array([1])),
            "pred5": rbst.Predicate("pred5", np.array([1]), np.array([-1])),
            "pred6": rbst.Predicate("pred6", np.array([-1]), np.array([1])),
            "pred7": rbst.Predicate("pred7", np.array([-1]), np.array([-1])),
            "pred8": rbst.Predicate("pred8", np.array([1]), np.array([110000])),
            "pred9": rbst.Predicate("pred9", np.array([1]), np.array([0.0000112])),"""

        """cls._vars = [
            "pred1",
            "pred2",
            "pred3",
            "pred4",
            "pred5",
            "pred6",
            "pred7",
            "pred8",
            "pred9",
        ]"""