from parser.parser import parse
from _parser import ParserTestCase
import stgem.objective.Robustness as rbst
class NextTestCase(ParserTestCase):
    def assert_is_next(self, value):
        self.assertIsInstance(value, rbst.Next)

    def test_next(self) -> None:
        self.assert_is_next(parse(r"next pred1", self._preds))

    def test_next_alternate_synax(self) -> None:
        self.assert_is_next(parse(r"next X pred1", self._preds))

    def test_next_with_and(self) -> None:
        self.assert_is_next(parse(r"next (pred1 && pred2)", self._preds))

    def test_nested_next_with_and(self) -> None:
        self.assert_is_next(parse(r"X (pred1 && X X X pred3)", self._preds))

    def test_many_next(self) -> None:
        self.assert_is_next(parse(r"X X X X X X X X (pred1 & pred2)", self._preds))
