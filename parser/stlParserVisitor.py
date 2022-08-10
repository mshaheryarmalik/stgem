# Generated from stlParser.g4 by ANTLR 4.5.1
from antlr4 import *

if __name__ is not None and "." in __name__:
    from .stlParser import stlParser
else:
    from stlParser import stlParser

from stgem.objective import Robustness as rbst
import numpy as np

# This class defines a complete generic visitor for a parse tree produced by stlParser.
class stlParserVisitor(ParseTreeVisitor):
    def __init__(self, lexer, predicates, mode):
        if not isinstance(predicates, (dict, list, tuple)):
            raise ValueError("predicates must be list, dict or tuple")

        if isinstance(predicates, dict):
            if not all(isinstance(pred, rbst.Predicate) for pred in predicates.values()):
                raise ValueError("all dictionary values must be TLTK Predicate objects")

        self._lexer = lexer
        self._mode = mode
        self._predicates = predicates

    # Visit a parse tree produced by stlParser#stlSpecification.
    def visitStlSpecification(self, ctx: stlParser.StlSpecificationContext):
        return self.visit(ctx.getRuleContext().getChild(0))

    # Visit a parse tree produced by stlParser#predicateExpr.
    def visitPredicateExpr(self, ctx: stlParser.PredicateExprContext):
        return self.visit(ctx.getRuleContext().getChild(0))

    # Visit a parse tree produced by stlParser#opFutureExpr.
    def visitOpFutureExpr(self, ctx: stlParser.OpFutureExprContext):
        pred = self.visit(ctx.getRuleContext().getChild(2))
        bounds = self.visit(ctx.getRuleContext().getChild(1))

        return rbst.Finally(bounds[0], bounds[1], pred, self._mode)

    # Visit a parse tree produced by stlParser#parenPhiExpr.
    def visitParenPhiExpr(self, ctx: stlParser.ParenPhiExprContext):
        return self.visit(ctx.getRuleContext().getChild(1))

    # Visit a parse tree produced by stlParser#opUntilExpr.
    def visitOpUntilExpr(self, ctx: stlParser.OpUntilExprContext):
        pred1 = self.visit(ctx.getRuleContext().getChild(0))
        pred2 = self.visit(ctx.getRuleContext().getChild(3))

        bounds = self.visit(ctx.getRuleContext().getChild(2))

        return rbst.Until(bounds[0], bounds[1], pred1, pred2, self._mode)

    # Visit a parse tree produced by stlParser#opGloballyExpr.
    def visitOpGloballyExpr(self, ctx: stlParser.OpGloballyExprContext):
        pred = self.visit(ctx.getRuleContext().getChild(2))
        bounds = self.visit(ctx.getRuleContext().getChild(1))

        return rbst.Global(bounds[0], bounds[1], pred, self._mode)

    # Visit a parse tree produced by stlParser#opLogicalExpr.
    def visitOpLogicalExpr(self, ctx: stlParser.OpLogicalExprContext):
        pred1 = self.visit(ctx.getRuleContext().getChild(0))
        pred2 = self.visit(ctx.getRuleContext().getChild(2))

        type = ctx.getRuleContext().getChild(1).getSymbol().type

        if type == self._lexer.ANDOP:
            return rbst.And(pred1, pred2, self._mode)
        elif type == self._lexer.OROP:
            return rbst.Or(pred1, pred2, self._mode)

    # Visit a parse tree produced by stlParser#opReleaseExpr.
    def visitOpReleaseExpr(self, ctx: stlParser.OpReleaseExprContext):
        pred1 = self.visit(ctx.getRuleContext().getChild(0))
        pred2 = self.visit(ctx.getRuleContext().getChild(3))

        bounds = self.visit(ctx.getRuleContext().getChild(2))

        return rbst.Not(rbst.Until(bounds[0], bounds[1], rbst.Not(pred1), rbst.Not(pred2)))

    # Visit a parse tree produced by stlParser#opNextExpr.
    def visitOpNextExpr(self, ctx: stlParser.OpNextExprContext):
        pred = self.visit(ctx.getRuleContext().getChild(1))

        return rbst.Next(pred)

    # Visit a parse tree produced by stlParser#opPropExpr.
    def visitOpPropExpr(self, ctx: stlParser.OpPropExprContext):
        pred1 = self.visit(ctx.getRuleContext().getChild(0))
        pred2 = self.visit(ctx.getRuleContext().getChild(2))

        return rbst.Or(rbst.Not(pred1), pred2, self._mode)

    # Visit a parse tree produced by stlParser#opNegExpr.
    def visitOpNegExpr(self, ctx: stlParser.OpNegExprContext):
        pred = self.visit(ctx.getRuleContext().getChild(1))

        return rbst.Not(pred, self._mode)

    # Visit a parse tree produced by stlParser#predicate.
    def visitPredicate(self, ctx: stlParser.PredicateContext):
        if ctx.getRuleContext().getChildCount() == 1:
            child_name = ctx.getRuleContext().getChild(0).getText()

            # the predicate name only exists, so return the relevant
            # rbst.Predicate data structure
            if not isinstance(self._predicates, dict):
                raise ValueError(
                    "singular variable names require predicates be provided as dictionary"
                )

            return self._predicates[child_name]
        elif True: # TODO:REQUIRES CONDITION
            var1: str = ctx.getRuleContext().getChild(0).getText()
            operator: str = ctx.getRuleContext().getChild(1).getText()
            var2: str = ctx.getRuleContext().getChild(2).getText()
            relop: str = ctx.getRuleContext().getChild(3).getText()
            value: str = ctx.getRuleContext().getChild(4).getText()

            if operator == "+":
                difference = rbst.Sum(var1, var2)
            else:
                difference = rbst.Subtract(var1, var2)

            #TODO:SORT OUT RETURN
            if operator == "<":
                rbst.LessThan(difference, value)

            elif operator == ">":
                raise Exception(
                    "Warning: strict relational operator used. Please use non-strict (<= or >=)"
                )
            elif operator == "<=":
                return rbst.Predicate(var, 1, float(value))
            elif operator == ">=":
                return rbst.Predicate(var, -1, -float(value))
        else:
            minus: str = ctx.getRuleContext().getChild(0).getText()

            if minus == "-":
                var: str = ctx.getRuleContext().getChild(1).getText()
                operator: str = ctx.getRuleContext().getChild(2).getText()
                value: str = ctx.getRuleContext().getChild(3).getText()
            else:
                var: str = ctx.getRuleContext().getChild(0).getText()
                operator: str = ctx.getRuleContext().getChild(1).getText()
                value: str = ctx.getRuleContext().getChild(2).getText()

            # check that the variable is valid
            if var not in self._predicates:
                raise Exception(
                    f"Error: predicate {var} is not in the list of valid variables {self._predicates}"
                )

            if operator == "<" or operator == ">":
                raise Exception(
                    "Warning: strict relational operator used. Please use non-strict (<= or >=)"
                )
            elif operator == "<=":
                if minus == "-":
                    return rbst.Predicate(var, -1, float(value))
                else:
                    return rbst.Predicate(var, 1, float(value))
            elif operator == ">=":
                if minus == "-":
                    return rbst.Predicate(var, 1, -float(value))
                else:
                    return rbst.Predicate(var, -1, -float(value))

    # Visit a parse tree produced by stlParser#interval.
    def visitInterval(self, ctx: stlParser.IntervalContext):
        bounds = []
        bounds.append(float(ctx.getRuleContext().getChild(1).getText()))
        bounds.append(float(ctx.getRuleContext().getChild(3).getText()))

        return bounds
