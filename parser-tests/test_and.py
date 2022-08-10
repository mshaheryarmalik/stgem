from unittest import skipIf
_can_parse = True
import sys
print(sys.path)
from parser.parser import parse
from _parser import ParserTestCase


@skipIf(_can_parse is False, "And parsing test case is not available without parsing functionality")
class AndTestCase(ParserTestCase):
    def assert_is_and(self, value):
        import tltk_mtl as mtl

        self.assertIsInstance(value, mtl.And)
    def test_logical_and(self) -> None:
        self.assert_is_and(parse(r"pred1 && pred2", self._preds))

    def test_logical_and_alternate_syntax(self) -> None:
        self.assert_is_and(parse(r"pred1 & pred1 && pred1 /\ pred1 and pred1", self._preds))

    def test_logical_and_with_negation(self) -> None:
        self.assert_is_and(parse(r"pred1 && !pred2 && pred3", self._preds))

    def test_logical_and_with_or(self) -> None:
        self.assert_is_and(parse(r"pred1 || pred2 && pred3", self._preds))

    def logical_and_with_or_alternate_syntax(self) -> None:
        self.assert_is_and(parse(r"pred2 | pred3 && pred2 | pred2 && pred1", self._preds))

