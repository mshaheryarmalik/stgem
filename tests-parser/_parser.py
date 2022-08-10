from unittest import SkipTest, TestCase
from stgem.objective import Robustness as rbst
import numpy as np


class ParserTestCase(TestCase):
    @classmethod
    def setUpClass(cls) -> None:

        cls._preds = {
            "pred1": rbst.Predicate("pred1", np.array([1]), np.array([2])),
            "pred2": rbst.Predicate("pred2", np.array([1]), np.array([4])),
            "pred3": rbst.Predicate("pred3", np.array([1]), np.array([8])),
            "pred4": rbst.Predicate("pred4", np.array([1]), np.array([1])),
            "pred5": rbst.Predicate("pred5", np.array([1]), np.array([-1])),
            "pred6": rbst.Predicate("pred6", np.array([-1]), np.array([1])),
            "pred7": rbst.Predicate("pred7", np.array([-1]), np.array([-1])),
            "pred8": rbst.Predicate("pred8", np.array([1]), np.array([110000])),
            "pred9": rbst.Predicate("pred9", np.array([1]), np.array([0.0000112])),
        }

        cls._vars = [
            "pred1",
            "pred2",
            "pred3",
            "pred4",
            "pred5",
            "pred6",
            "pred7",
            "pred8",
            "pred9",
        ]