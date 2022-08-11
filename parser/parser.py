from __future__ import annotations

from typing import Dict, Optional, Sequence, Union
from stgem.objective import Robustness as rbst
from antlr4.CommonTokenStream import CommonTokenStream
from antlr4.InputStream import InputStream

from .stlLexer import stlLexer as Lexer
from .stlParser import stlParser as Parser
from .visitor import Visitor
PredicateNameSeq = Sequence[str]
PredicateDict = Dict[str, rbst.Predicate]
Predicates = Union[PredicateNameSeq, PredicateDict]
TltkObject = Union[
    rbst.And,
    rbst.Finally,
    rbst.Global,
    rbst.Implication,
    rbst.Next,
    rbst.Not,
    rbst.Or,
    rbst.Predicate, #not implemented
]


def parse(formula: str, predicates: Predicates, mode: str = "cpu") -> Optional[TltkObject]:
    """TLTk parser parses a specification requirement into an equivalent TLTk structure

    Attributes:
        formula: The formal specification requirement
        predicates: The set of Predicate(s) used in the requirement
        mode: The TLTk computation mode
    """
    input_stream = InputStream(formula)

    lexer = Lexer(input_stream)
    stream = CommonTokenStream(lexer)

    parser = Parser(stream)
    tree = parser.stlSpecification()
    visitor = Visitor(lexer, predicates, mode)

    return visitor.visit(tree)  # type: ignore
