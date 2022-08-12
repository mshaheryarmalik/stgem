# Generated from stlParser.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .stlParser import stlParser
else:
    from stlParser import stlParser

from stgem.objective.Robustness import *

# This class defines a complete generic visitor for a parse tree produced by stlParser.

class stlParserVisitor(ParseTreeVisitor):

    def __init__(self, predicates):
        self.predicates = predicates

# Ovveride this function to ensure that TerminalNodeImpl does not ovveride the returned value with a 'None'
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


    # Visit a parse tree produced by stlParser#opFutureExpr.
    def visitOpFutureExpr(self, ctx:stlParser.OpFutureExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpFutureExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#parenPhiExpr.
    def visitParenPhiExpr(self, ctx:stlParser.ParenPhiExprContext):
        print("-----------------------------------{}------------------------------------------".format("visitParenPhiExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opUntilExpr.
    def visitOpUntilExpr(self, ctx:stlParser.OpUntilExprContext):
        print("-----------------------------------{}------------------------------------------".format("vOpUntilExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opGloballyExpr.
    def visitOpGloballyExpr(self, ctx:stlParser.OpGloballyExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpGloballyExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opLogicalExpr.
    def visitOpLogicalExpr(self, ctx:stlParser.OpLogicalExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpLogicalExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opReleaseExpr.
    def visitOpReleaseExpr(self, ctx:stlParser.OpReleaseExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpReleaseExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opNextExpr.
    def visitOpNextExpr(self, ctx:stlParser.OpNextExprContext):
        print("-----------------------------------{}------------------------------------------".format("visitOpNextExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opPropExpr.
    def visitOpPropExpr(self, ctx:stlParser.OpPropExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpPropExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#opNegExpr.
    def visitOpNegExpr(self, ctx:stlParser.OpNegExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpNegExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#predNumber.
    def visitPredNumber(self, ctx:stlParser.PredNumberContext):
        print("-----------------------------------{}------------------------------------------".format("PredNumber"))
        value = float(ctx.getText())
        return Const(value)


    # Visit a parse tree produced by stlParser#predName.
    def visitPredName(self, ctx:stlParser.PredNameContext):
        print("-----------------------------------{}------------------------------------------".format("PredName"))
        name = ctx.getRuleContext().getChild(0).getText()
        return self.predicates[name]


    # Visit a parse tree produced by stlParser#predArithLeftExpr.
    def visitPredArithLeftExpr(self, ctx:stlParser.PredArithLeftExprContext):
        print("-----------------------------------{}------------------------------------------".format("PredArithLeftExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#predArithRightExpr.
    def visitPredArithRightExpr(self, ctx:stlParser.PredArithRightExprContext):
        print("-----------------------------------{}------------------------------------------".format("PredArithRightExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#predRelExpr.
    def visitPredRelExpr(self, ctx:stlParser.PredRelExprContext):
        print("-----------------------------------{}------------------------------------------".format("PredRealExpr"))
        return self.visitChildren(ctx)


    # Visit a parse tree produced by stlParser#interval.
    def visitInterval(self, ctx:stlParser.IntervalContext):
        print("-----------------------------------{}------------------------------------------".format("Interval"))
        bounds = []
        bounds.append(float(ctx.getRuleContext().getChild(1).getText()))
        bounds.append(float(ctx.getRuleContext().getChild(3).getText()))
        return bounds


del stlParser
