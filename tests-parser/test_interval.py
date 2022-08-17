from parser.parser import parse
from _parser import ParserTestCase

class IntervalTestCase(ParserTestCase):

    def _do_test(self, phi, type):
        """Base tester method"""
        spec = parse(phi)
        self.assertIsInstance(spec, type)

    # TODO: Never visits this for some reason
    def test_interval(self):
        self._do_test("{}".format(self._interval), type(list()))