from parser.parser import parse
from _parser import ParserTestCase
import stgem.objective.Robustness as rbst

class PredicateTestCase(ParserTestCase):

    def test_number(self):
        assert isinstance(parse(r"5", [rbst.Signal(r"name", (-1,1))]), rbst.Const)

    def test_name(self):
        assert isinstance(parse(r"name", {"name":rbst.Signal(r"name", (-1,1))}), rbst.Signal)