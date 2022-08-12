from __future__ import annotations

from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream

from .stlLexer import stlLexer as Lexer
from .stlParser import stlParser as Parser
from .visitor import stlParserVisitor as Visitor
import stgem.objective.Robustness as rbst

def parse(formula: str, predicates: rbst.Predicate):
    """TLTk parser parses a specification requirement into an equivalent TLTk structure

    Attributes:
        formula: The formal specification requirement
        predicates: The set of Predicate(s) used in the requirement
    """
    input_stream = InputStream(formula)

    lexer = Lexer(input_stream)
    stream = CommonTokenStream(lexer)

    parser = Parser(stream)
    tree = parser.stlSpecification()
    visitor = Visitor(predicates)

    return visitor.visit(tree)  # type: ignore
