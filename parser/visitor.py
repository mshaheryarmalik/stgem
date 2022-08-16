# Generated from stlParser.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .stlParser import stlParser
else:
    from stlParser import stlParser

from stgem.objective.Robustness import *

# This class defines a complete generic visitor for a parse tree produced by stlParser.

class stlParserVisitor(ParseTreeVisitor):

# Override this function to ensure that TerminalNodeImpl does not override the returned value with a 'None'
    def aggregateResult(self, aggregate, nextResult):
        if not aggregate is None:
            return aggregate
        else:
            return nextResult


    # Visit a parse tree produced by stlParser#stlSpecification.
    def visitStlSpecification(self, ctx:stlParser.StlSpecificationContext):
        print("-----------------------------------{}------------------------------------------".format("StlSpecification"))
        value = self.visitChildren(ctx)
        print("StlSpecification", value)
        return value


    # Visit a parse tree produced by stlParser#predicateExpr.
    def visitPredicateExpr(self, ctx:stlParser.PredicateExprContext):
        print("-----------------------------------{}------------------------------------------".format("PredicateExpr"))
        value = self.visitChildren(ctx)
        print("PredicateExpr", value)
        return value


    # Visit a parse tree produced by stlParser#signalExpr.
    def visitSignalExpr(self, ctx:stlParser.SignalExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opFutureExpr.
    def visitOpFutureExpr(self, ctx:stlParser.OpFutureExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#parenPhiExpr.
    def visitParenPhiExpr(self, ctx:stlParser.ParenPhiExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opUntilExpr.
    def visitOpUntilExpr(self, ctx:stlParser.OpUntilExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opGloballyExpr.
    def visitOpGloballyExpr(self, ctx:stlParser.OpGloballyExprContext):
        print("-----------------------------------{}------------------------------------------".format("GloballyExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opLogicalExpr.
    def visitOpLogicalExpr(self, ctx:stlParser.OpLogicalExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opReleaseExpr.
    def visitOpReleaseExpr(self, ctx:stlParser.OpReleaseExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opNextExpr.
    def visitOpNextExpr(self, ctx:stlParser.OpNextExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opPropExpr.
    def visitOpPropExpr(self, ctx:stlParser.OpPropExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opNegExpr.
    def visitOpNegExpr(self, ctx:stlParser.OpNegExprContext):
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#signalParenthesisExpr.
    def visitSignalParenthesisExpr(self, ctx:stlParser.SignalParenthesisExprContext):
        print("-----------------------------------{}------------------------------------------".format("SignalParenthesisExpr"))
        return ctx.getRuleContext().getChild(1).getText()


    # Visit a parse tree produced by stlParser#signalName.
    def visitSignalName(self, ctx:stlParser.SignalNameContext):
        print("-----------------------------------{}------------------------------------------".format("SignalName"))
        name = ctx.getText()
        return Signal(name)


    # Visit a parse tree produced by stlParser#signalUnaryExpr.
    def visitSignalUnaryExpr(self, ctx:stlParser.SignalUnaryExprContext):
        print("-----------------------------------{}------------------------------------------".format("SignalUnaryExpr"))
        operator = ctx.getRuleContext().getChild(0).getText()
        signal = self.visit(ctx.getRuleContext().getChild(1))
        if operator == "+":
            return signal
        elif operator == "-":
            return Mult(Constant(-1), signal)

    # Visit a parse tree produced by stlParser#signalSumExpr.
    def visitSignalSumExpr(self, ctx:stlParser.SignalSumExprContext):
        print("-----------------------------------{}------------------------------------------".format("SignalSumExpr"))
        signal1 = self.visit(ctx.getRuleContext().getChild(0))
        operator = ctx.getRuleContext().getChild(1).getText()
        signal2 = self.visit(ctx.getRuleContext().getChild(2))
        if operator == "+":
            return Sum(signal1, signal2)
        elif operator == "-":
            return Subtract(signal1, signal2)


    # Visit a parse tree produced by stlParser#signalNumber.
    def visitSignalNumber(self, ctx:stlParser.SignalNumberContext):
        print("-----------------------------------{}------------------------------------------".format("SignalNumber"))
        value = float(ctx.getText())
        return Constant(value)


    # Visit a parse tree produced by stlParser#signalMultExpr.
    def visitSignalMultExpr(self, ctx:stlParser.SignalMultExprContext):
        print("-----------------------------------{}------------------------------------------".format("SignalMultExpr"))
        signal1 = self.visit(ctx.getRuleContext().getChild(0))
        operator = ctx.getRuleContext().getChild(1).getText()
        signal2 = self.visit(ctx.getRuleContext().getChild(2))
        if operator == "*":
            return Mult(signal1, signal2)
        elif operator == "/":
            pass
            # TODO: add division when implemented in Robustness
            #return Div?(signal1, signal2)


    # Visit a parse tree produced by stlParser#interval.
    def visitInterval(self, ctx:stlParser.IntervalContext):
        print("-----------------------------------{}------------------------------------------".format("Interval"))
        bounds = []
        bounds.append(float(ctx.getRuleContext().getChild(1).getText()))
        bounds.append(float(ctx.getRuleContext().getChild(3).getText()))
        return bounds

del stlParser
