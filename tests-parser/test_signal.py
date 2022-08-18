from parser.parser import parse
from _parser import ParserTestCase
from stgem.objective.Robustness import *
import numpy as np

class SignalTestCase(ParserTestCase):

    def _do_test(self, phi, expected, type):
        """Base tester method"""
        spec = parse(phi)
        result = spec.eval(self._traces)
        self.assertIsInstance(spec, type)
        np.testing.assert_equal(result, expected)

    def test_number(self):
        num = 5
        self._do_test("{}".format(num), np.full(len(self._timestamps), num), Constant)

    def test_name(self):
        key = "signal1"
        self._do_test("{}".format(key), self._signals[key], Signal)

    def test_signalParenthesisExpr(self):
        key = "signal1"
        num = 5
        # Test with Signal
        self._do_test("({})".format(key), self._signals[key], Signal)
        # Test with Constant
        self._do_test("({})".format(num), np.full(len(self._timestamps), num), Constant)

    def _do_test_operator(self, keys, nums, operators, types, funcs):
        """Operator tester method """
        # Test Signals x Constants
        for key in keys:
            for num in nums:
                for i in range(len(operators)):
                    print(operators[i])
                    self._do_test("{}{}{}".format(key, operators[i], num),
                                  funcs[i](self._signals[key], np.full(len(self._timestamps), num)),
                                  types[i])
            # Test Signal x Signal
            for i in range(len(operators)):
                print(operators[i])
                self._do_test("{}{}{}".format(key, operators[i], keys[keys.index(key) - 1]),
                              funcs[i](self._signals[key], self._signals[keys[keys.index(key) - 1]]),
                              types[i])
        # Test Constants x Signals
        for num in nums:
            for key in keys:
                for i in range(len(operators)):
                    print(operators[i])
                    self._do_test("{}{}{}".format(num, operators[i], key),
                                  funcs[i](np.full(len(self._timestamps), num), self._signals[key]),
                                  types[i])
            # Test Constant x Constant
            for i in range(len(operators)):
                print(operators[i])
                self._do_test("{}{}{}".format(num, operators[i], nums[nums.index(num) - 1]),
                              funcs[i](np.full(len(self._timestamps), num), np.full(len(self._timestamps), nums[nums.index(num) - 1])),
                              types[i])

    def test_signalMultExpr(self):
        keys = ["signal1", "signal2"]
        nums = [5, 10]
        operators = ["*", "/"]
        types = [Mult, object] # TODO replace object with Div
        funcs = [np.multiply, np.divide]
        self._do_test_operator(keys, nums, operators, types, funcs)

    def test_signalSumExpr(self):
        keys = ["signal1", "signal2"]
        nums = [5, 10]
        operators = ["+", "- "] # NOTE space after '-' as otherwise following constants would be interpreted as negative
        types = [Sum, Subtract]
        funcs = [np.add, np.subtract]
        self._do_test_operator(keys, nums, operators, types, funcs)