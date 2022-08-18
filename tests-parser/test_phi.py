from stl.parser import parse
from _parser import ParserTestCase
from stl.robustness import *

class PhiTestCase(ParserTestCase):

    def _do_test(self, phi, type):
        """Base tester method. Only checks for correct class"""
        spec = parse(phi)
        self.assertIsInstance(spec, type)

    def test_phiParenthesisExpr(self):
        key = "signal1"
        self._do_test("({})".format(key), Signal)

    def test_absPhiExpr(self):
        key = "signal1"
        self._do_test("|{}|".format(key), Abs)

    def test_opNegExpr(self):
        key = "signal1"
        operators = ['~', '!', 'not '] # NOTE space after 'not'
        for op in operators:
            self._do_test("{}{}".format(op, key), Not)

    def test_opNextExpr(self):
        key = "signal1"
        operators = ['next ', 'X '] # NOTE space after operators
        suffixes = ['_', '']
        for op in operators:
            for suff in suffixes:
                    self._do_test("{}{}{}".format(op, suff, key), Next)

    def test_opFutureExpr(self):
        key = "signal1"
        operators = [' finally ', ' eventually ', ' F ', '<>']  # NOTE space surrounding operators (except '<>')
        suffixes = [self._interval, '']
        for op in operators:
            for suff in suffixes:
                self._do_test("{}{}{}".format(op, suff, key), Finally)

    def test_opGloballyExpr(self):
        key = "signal1"
        operators = [' globally ', ' always ', ' G ', '[]'] # NOTE space surrounding operators (except '[]')
        suffixes = [self._interval, '']
        for op in operators:
            for suff in suffixes:
                self._do_test("{}{}{}".format(op, suff, key), Global)

    def test_opUntilExpr(self):
        key = "signal1"
        operators = [' until ', ' U '] # NOTE space surrounding operators
        suffixes = [self._interval, '']
        for op in operators:
            for suff in suffixes:
                self._do_test("{}{}{}{}".format(key, op, suff, key), object) # TODO replace object with Until

    def test_logicalExpr(self):
        key = "signal1"
        andOperators = [' and '] # NOTE space surrounding 'and'
        orOperators = [' or '] # NOTE space surrounding 'or'
        # Test and
        for op in andOperators:
            self._do_test("{}{}{}".format(key, op, key), And)
        # Test or
        for op in orOperators:
            print("OPERATOR")
            print(op)
            self._do_test("{}{}{}".format(key, op, key), Or)

    def test_opPropExpr(self):
        key = "signal1"
        impliesOperators = [' implies ', '->'] # NOTE space surrounding 'implies'
        equivOperators = [' iff ', '<->'] # NOTE space surrounding 'iff'
        # Test implication
        for op in impliesOperators:
            self._do_test("{}{}{}".format(key, op, key), Implication)
        # Test equivalent
        for op in equivOperators:
            self._do_test("{}{}{}".format(key, op, key), Equals)

    def test_predicateExpr(self):
        key = "signal1"
        operators = ['<=', '>=', '<', '>']
        types = [LessThan, GreaterThan, StrictlyLessThan, StrictlyGreaterThan]
        for i, op in enumerate(operators):
            self._do_test("{}{}{}".format(key, op, key), types[i])