from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream

from stl.stlLexer import stlLexer as Lexer
from stl.stlParser import stlParser as Parser
from stl.visitor import stlParserVisitor as Visitor

def parse(phi, ranges=None, nu=None):
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
    visitor.ranges = ranges
    visitor.nu = nu

    return visitor.visit(tree)  # type: ignore

