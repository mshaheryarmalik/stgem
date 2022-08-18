import numpy as np
from unittest import TestCase
import stl.robustness as rbst

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