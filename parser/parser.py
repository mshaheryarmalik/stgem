from __future__ import annotations

from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream

from .stlLexer import stlLexer as Lexer
from .stlParser import stlParser as Parser
from .visitor import stlParserVisitor as Visitor

def parse(phi):
    """ parses a specification requirement into an equivalent STL structure

    Attributes:
        formula: The formal specification requirement
        signals: The set of Predicate(s) used in the requirement
        timestamps:
    """
    input_stream = InputStream(phi)

    lexer = Lexer(input_stream)
    stream = CommonTokenStream(lexer)

    parser = Parser(stream)
    tree = parser.stlSpecification()
    visitor = Visitor()

    return visitor.visit(tree)  # type: ignore
