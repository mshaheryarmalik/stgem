from parser.parser import parse
from _parser import ParserTestCase
import stgem.objective.Robustness as rbst

class SignalTestCase(ParserTestCase):

    def _do_test_number(self, phi, expected_number, key):
        result = parse(phi, self._timestamps[key], self._signals)
        self.assertIsInstance(result, rbst.Constant)
        self.assertEqual(result.val, expected_number)

    def _do_test_name(self, phi, expected_range, key):
        result = parse(phi, self._timestamps[key], self._signals)
        self.assertIsInstance(result, rbst.Signal)
        self.assertEqual(result.name, key)
        self.assertEqual(result.range, expected_range)

    def range(self, key1, key2=None):
        if key2 == None:
            return [min(self._signals[key1]), max(self._signals[key1])]
        else:
            return [min(min(self._signals[key1]), min(self._signals[key2])), max(max(self._signals[key1]), max(self._signals[key2]))]

    def test_number(self):
        key = "signal1"
        num = 5
        self._do_test_number("{}".format(num), num, key)

    def test_name(self):
        key = "signal1"
        self._do_test_name(key, self.range(key), key)

    def test_signalParenthesisExpr(self):
        key = "signal1"
        num = 5
        # Test with Signal
        self._do_test_name("({})".format(key), self.range(key), key)
        # Test with Constant
        self._do_test_number("({})".format(num), num, key)

    # TODO: Visitor does not visit unary expressions at all
    """def test_signalUnaryExpr(self):
        key = "signal1"
        num = 5
        # Test with Signal
        self._do_test_name("+"+key, self.range(key), key)
        self._do_test_name("-"+key, [-x for x in self.range(key)], key)
        # Test with Constant
        self._do_test_number("+{}".format(num), num, key)
        self._do_test_number("-{}".format(num), -num, key)"""

    # TODO:
    def test_signalMultExpr(self):
        pass

    # TODO:
    def test_signalSumExpr(self):
        keys = ["signal1", "signal2"]
        nums = [5, 10]
        for key in keys:
            for num in nums:
                pass