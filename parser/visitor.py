# Generated from stlParser.g4 by ANTLR 4.10.1
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .stlParser import stlParser
else:
    from stlParser import stlParser

from stgem.objective.Robustness import *

# This class defines a complete generic visitor for a parse tree produced by stlParser.

class stlParserVisitor(ParseTreeVisitor):

    # Visit a parse tree produced by stlParser#stlSpecification.
    def visitStlSpecification(self, ctx:stlParser.StlSpecificationContext):
        return self.visit(ctx.getRuleContext().getChild(0))


    # Visit a parse tree produced by stlParser#predicateExpr.
    def visitPredicateExpr(self, ctx:stlParser.PredicateExprContext):
        phi1 = self.visit(ctx.getRuleContext().getChild(0))
        operator = ctx.getRuleContext().getChild(1).getText()
        phi2 = self.visit(ctx.getRuleContext().getChild(2))
        if operator == "<=":
            return LessThan(phi1, phi2)
        elif operator == ">=":
            return GreaterThan(phi1, phi2)
        elif operator == "<":
            raise NotImplementedError("Strict < not implemented.")
        else:
            raise NotImplementedError("Strict > not implemented.")


    # Visit a parse tree produced by stlParser#signalExpr.
    def visitSignalExpr(self, ctx:stlParser.SignalExprContext):
        return self.visit(ctx.getRuleContext().getChild(0))


    # Visit a parse tree produced by stlParser#opFutureExpr.
    def visitOpFutureExpr(self, ctx:stlParser.OpFutureExprContext):
        if ctx.getRuleContext().getChildCount() == 2:
            raise NotImplementedError("Eventually not supported without specifying an interval.")
        elif ctx.getRuleContext().getChildCount() == 3:
            phi = self.visit(ctx.getRuleContext().getChild(2))
            interval = self.visit(ctx.getRuleContext().getChild(1))
        return Finally(interval[0], interval[1], phi)


    # Visit a parse tree produced by stlParser#parenPhiExpr.
    def visitParenPhiExpr(self, ctx:stlParser.ParenPhiExprContext):
        return self.visit(ctx.getRuleContext().getChild(1))


    # Visit a parse tree produced by stlParser#absPhiExpr.
    def visitAbsPhiExpr(self, ctx:stlParser.AbsPhiExprContext):
        return Abs(self.visit(ctx.getRuleContext().getChild(1)))


    # Visit a parse tree produced by stlParser#opUntilExpr.
    def visitOpUntilExpr(self, ctx:stlParser.OpUntilExprContext):
        phi1 = self.visit(ctx.getRuleContext().getChild(0))
        if ctx.getRuleContext().getChildCount() == 3:
            raise NotImplementedError("Until not supported without specifying an interval.")
        elif ctx.getRuleContext().getChildCount() == 4: # Optional interval
            phi2 = self.visit(ctx.getRuleContext().getChild(3))
            interval = self.visit(ctx.getRuleContext().getChild(2))
            raise NotImplementedError("Until not implemented.")


    # Visit a parse tree produced by stlParser#opGloballyExpr.
    def visitOpGloballyExpr(self, ctx:stlParser.OpGloballyExprContext):
        if ctx.getRuleContext().getChildCount() == 2:
            raise NotImplementedError("Global not supported without specifying an interval.")
        elif ctx.getRuleContext().getChildCount() == 3:
            phi = self.visit(ctx.getRuleContext().getChild(2))
            interval = self.visit(ctx.getRuleContext().getChild(1))
        return Global(interval[0], interval[1], phi)


    # Visit a parse tree produced by stlParser#opLogicalExpr.
    def visitOpLogicalExpr(self, ctx:stlParser.OpLogicalExprContext):
        phi1 = self.visit(ctx.getRuleContext().getChild(0))
        operator = ctx.getRuleContext().getChild(1).getText()
        phi2 = self.visit(ctx.getRuleContext().getChild(2))
        if operator in ["and"]:
            return And(phi1, phi2)
        else:
            return Or(phi1, phi2)


    # Visit a parse tree produced by stlParser#opNextExpr.
    def visitOpNextExpr(self, ctx:stlParser.OpNextExprContext):
        return Next(self.visit(ctx.getRuleContext().getChild(1)))


    # Visit a parse tree produced by stlParser#opPropExpr.
    def visitOpPropExpr(self, ctx:stlParser.OpPropExprContext):
        phi1 = self.visit(ctx.getRuleContext().getChild(0))
        operator = ctx.getRuleContext().getChild(1).getText()
        phi2 = self.visit(ctx.getRuleContext().getChild(2))
        if operator in ["implies", "->"]:
            return Implication(phi1, phi2)
        elif operator in ["iff", "<->"]:
            raise NotImplementedError("Equivalence not implemented.")


    # Visit a parse tree produced by stlParser#opNegExpr.
    def visitOpNegExpr(self, ctx:stlParser.OpNegExprContext):
        phi = self.visit(ctx.getRuleContext().getChild(1))
        return Not(phi)


    # Visit a parse tree produced by stlParser#signalParenthesisExpr.
    def visitSignalParenthesisExpr(self, ctx:stlParser.SignalParenthesisExprContext):
        return self.visit(ctx.getRuleContext().getChild(1))


    # Visit a parse tree produced by stlParser#signalName.
    def visitSignalName(self, ctx:stlParser.SignalNameContext):
        name = ctx.getText()
        return Signal(name)


    # Visit a parse tree produced by stlParser#signalSumExpr.
    def visitSignalSumExpr(self, ctx:stlParser.SignalSumExprContext):
        signal1 = self.visit(ctx.getRuleContext().getChild(0))
        operator = ctx.getRuleContext().getChild(1).getText()
        signal2 = self.visit(ctx.getRuleContext().getChild(2))
        if operator == "+":
            return Sum(signal1, signal2)
        elif operator == "-":
            return Subtract(signal1, signal2)


    # Visit a parse tree produced by stlParser#signalNumber.
    def visitSignalNumber(self, ctx:stlParser.SignalNumberContext):
        value = float(ctx.getText())
        return Constant(value)


    # Visit a parse tree produced by stlParser#signalMultExpr.
    def visitSignalMultExpr(self, ctx:stlParser.SignalMultExprContext):
        signal1 = self.visit(ctx.getRuleContext().getChild(0))
        operator = ctx.getRuleContext().getChild(1).getText()
        signal2 = self.visit(ctx.getRuleContext().getChild(2))
        if operator == "*":
            return Mult(signal1, signal2)
        elif operator == "/":
            raise NotImplementedError


    # Visit a parse tree produced by stlParser#interval.
    def visitInterval(self, ctx:stlParser.IntervalContext):
        A = float(ctx.getRuleContext().getChild(1).getText())
        B = float(ctx.getRuleContext().getChild(1).getText())
        return [A, B]



del stlParser
