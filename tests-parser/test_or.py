from parser.parser import parse
from _parser import ParserTestCase
import stgem.objective.Robustness as rbst

class OrTestCase(ParserTestCase):
    def assert_is_or(self, value):
        self.assertIsInstance(value, rbst.Or)

    def test_or(self) -> None:
        self.assert_is_or(parse(r"pred1 || pred1", self._preds))

    def test_or_alternate_syntax(self) -> None:
        self.assert_is_or(parse(r"pred1 | pred1 || pred1 \/ pred1 or pred1", self._preds))

    def test_or_with_negation(self) -> None:
        self.assert_is_or(parse(r"pred1 or !pred2 || pred3", self._preds))

    def test_or_with_and(self) -> None:
        self.assert_is_or(parse(r"pred1 and pred2 or pred3", self._preds))

    def test_or_with_and_alternate_syntax(self) -> None:
        self.assert_is_or(parse(r"pred2 and pred3 & pred2 and pred2 or pred1", self._preds))
