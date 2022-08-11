from parser.parser import parse
from _parser import ParserTestCase
import stgem.objective.Robustness as rbst

class NegationTestCase(ParserTestCase):
    def assert_is_not(self, value):
        self.assertIsInstance(value, rbst.Not)

    def test_negation(self) -> None:
        self.assert_is_not(parse(r"not pred1", self._preds))
        self.assert_is_not(parse(r"!pred3", self._preds))

    def test_double_negation(self) -> None:
        self.assert_is_not(parse(r"!!(pred2)", self._preds))

    def test_negation_with_and(self) -> None:
        self.assert_is_not(parse(r"!(pred1 and pred2)", self._preds))

    def test_many_negation_with_and(self) -> None:
        self.assert_is_not(parse(r"!!!!!(pred1 and pred2 and pred3)", self._preds))
