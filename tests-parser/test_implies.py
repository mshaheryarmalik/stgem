from parser.parser import parse
from _parser import ParserTestCase
import stgem.objective.Robustness as rbst

class ImpliesTestCase(ParserTestCase):
    def assert_is_or(self, value):
        self.assertIsInstance(value, rbst.Or)

    def test_implication(self) -> None:
        self.assert_is_or(parse(r"pred1 -> pred1", self._preds))

    def test_implication_with_and(self) -> None:
        self.assert_is_or(parse(r"(pred1 && pred2) -> pred1", self._preds))

    def test_chained_implication(self) -> None:
        self.assert_is_or(parse(r"pred1 -> pred2 -> pred3", self._preds))

    def test_implication_with_multi_and(self) -> None:
        self.assert_is_or(parse(r"pred1 -> (pred1 && pred2 && pred3)", self._preds))

    def test_implication_with_or_and_literal(self) -> None:
        self.assert_is_or(parse(r"pred1 -> pred2 -> (pred1 || pred2) -> pred3", self._preds))
