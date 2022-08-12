from parser.parser import parse
from _parser import ParserTestCase

class IntervalTestCase(ParserTestCase):

    def test_interval(self) -> None:
        self.assertEqual(parse(r"(1, 100)"), (1, 100))