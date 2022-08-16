from parser.parser import parse
from _parser import ParserTestCase
from stgem.objective.Robustness import *
import numpy as np

class IntervalTestCase(ParserTestCase):

    def _do_test(self, phi, expected, type):
        """Base tester method"""
        spec = parse(phi)
        result = spec.eval(self._traces)
        self.assertIsInstance(spec, type)
        np.testing.assert_equal(result, expected)

    def test_interval(self):
        self._do_test("(1, 100)", [1, 100], list())