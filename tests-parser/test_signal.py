from parser.parser import parse
from _parser import ParserTestCase
import stgem.objective.Robustness as rbst
import numpy as np

class SignalTestCase(ParserTestCase):

    def test_number(self):
        key = "signal1"
        formula = "5"
        result = parse(formula, self._timestamps[key], self._signals)
        self.assertIsInstance(result, rbst.Constant)
        self.assertEqual(result.val, int(formula))

    def test_name(self):
        key = "signal1"
        formula = key
        result = parse(formula, self._timestamps[key], self._signals)
        self.assertIsInstance(result, rbst.Signal)
        self.assertEqual(result.name, key)
        self.assertEqual(result.range, [min(self._signals[key]), max(self._signals[key])])

    def test_signalParenthesisExpr(self):
        key = "signal1"
        formula = "("+key+")"
        result = parse(formula, self._timestamps[key], self._signals)
        self.assertIsInstance(result, rbst.Signal)
        self.assertEqual(result.name, key)
        self.assertEqual(result.range, [min(self._signals[key]), max(self._signals[key])])

    def test_signalUnaryExpr(self):
        pass

    def test_signalMultExpr(self):
        pass

    def test_signalSumExpr(self):
        pass