# Generated from stlParser.g4 by ANTLR 4.7.2
from antlr4 import *
if __name__ is not None and "." in __name__:
    from .stlParser import stlParser
else:
    from stlParser import stlParser

from stgem.objective.Robustness import *

# This class defines a complete generic visitor for a parse tree produced by stlParser.

class stlParserVisitor(ParseTreeVisitor):
    """All functions have a temporary debug print that indicates when they are called"""

    # Visit a parse tree produced by stlParser#stlSpecification.
    def visitStlSpecification(self, ctx:stlParser.StlSpecificationContext):
        print("-----------------------------------{}------------------------------------------".format("StlSpecification"))
        value = self.visit(ctx.getRuleContext().getChild(0))
        print("StlSpecification", value) # DEBUG
        return value


    # Visit a parse tree produced by stlParser#predicateExpr.
    def visitPredicateExpr(self, ctx:stlParser.PredicateExprContext):
        print("-----------------------------------{}------------------------------------------".format("PredicateExpr"))
        phi1 = self.visit(ctx.getRuleContext().getChild(0))
        operator = ctx.getRuleContext().getChild(1).getText()
        phi2 = self.visit(ctx.getRuleContext().getChild(2))
        if operator == "<=":
            return LessThan(phi1, phi2)
        elif operator == ">=":
            return GreaterThan(phi1, phi2)
        # TODO: Implement strict forms when implemented in Robustness
        '''elif operator == "<":
            return StrictlyLessThan(phi1, phi2)
        elif operator == ">":
            return StrictlyGreaterThan(phi1, phi2)'''


    # Visit a parse tree produced by stlParser#signalExpr.
    def visitSignalExpr(self, ctx:stlParser.SignalExprContext):
        print("-----------------------------------{}------------------------------------------".format("SignalExpr"))
        value = self.visit(ctx.getRuleContext().getChild(0))
        print("PredicateExpr", value) # DEBUG
        return value


    # Visit a parse tree produced by stlParser#opFutureExpr.
    def visitOpFutureExpr(self, ctx:stlParser.OpFutureExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpFutureExpr"))
        if ctx.getRuleContext().getChildCount() == 2:
            phi = self.visit(ctx.getRuleContext().getChild(1))
            # TODO: Finally currently requires an interval to function, but the parser accepts the operation without one.
            #       Implement a default interval?
            interval = [0, 1] # Temporary default interval
        elif ctx.getRuleContext().getChildCount() == 3: # Optional interval
            phi = self.visit(ctx.getRuleContext().getChild(2))
            interval = self.visit(ctx.getRuleContext().getChild(1))
        return Finally(interval[0], interval[1], phi)


    # Visit a parse tree produced by stlParser#parenPhiExpr.
    def visitParenPhiExpr(self, ctx:stlParser.ParenPhiExprContext):
        print("-----------------------------------{}------------------------------------------".format("ParenPhiExpr"))
        return self.visit(ctx.getRuleContext().getChild(1))


    # Visit a parse tree produced by stlParser#opUntilExpr.
    def visitOpUntilExpr(self, ctx:stlParser.OpUntilExprContext):
        print("-----------------------------------{}------------------------------------------".format("OPUntilExpr"))
        phi1 = self.visit(ctx.getRuleContext().getChild(0))
        if ctx.getRuleContext().getChildCount() == 3:
            phi2 = self.visit(ctx.getRuleContext().getChild(2))
            # TODO: Until will require(?) an interval to function, but the parser accepts the operation without one.
            #       Implement a default interval?
            interval = [0, 1]  # Temporary default interval
            #return Until(interval[0], interval[1], phi1, phi2)
        elif ctx.getRuleContext().getChildCount() == 4: # Optional interval
            phi2 = self.visit(ctx.getRuleContext().getChild(3))
            interval = self.visit(ctx.getRuleContext().getChild(2))
            # TODO: Use Until when implemented in Robustness
            #return Until(interval[0], interval[1], phi1, phi2)


    # Visit a parse tree produced by stlParser#opGloballyExpr.
    def visitOpGloballyExpr(self, ctx:stlParser.OpGloballyExprContext):
        print("-----------------------------------{}------------------------------------------".format("GloballyExpr"))
        if ctx.getRuleContext().getChildCount() == 2:
            phi = self.visit(ctx.getRuleContext().getChild(1))
            # TODO: Global currently requires an interval to function, but the parser accepts the operation without one.
            #       Implement a default interval?
            interval = [0, 1] # Temporary default interval
        elif ctx.getRuleContext().getChildCount() == 3: # Optional interval
            phi = self.visit(ctx.getRuleContext().getChild(2))
            interval = self.visit(ctx.getRuleContext().getChild(1))
        return Global(interval[0], interval[1], phi)


    # Visit a parse tree produced by stlParser#opLogicalExpr.
    def visitOpLogicalExpr(self, ctx:stlParser.OpLogicalExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpLogicalExpr"))
        phi1 = self.visit(ctx.getRuleContext().getChild(0))
        operator = ctx.getRuleContext().getChild(1).getText()
        phi2 = self.visit(ctx.getRuleContext().getChild(2))
        if operator in ['and', '/\\', '&&', '&']:
            return And(phi1, phi2)
        elif operator in ['or', '\\/', '||', '|']:
            return Or(phi1, phi2)


    # Visit a parse tree produced by stlParser#opReleaseExpr.
    def visitOpReleaseExpr(self, ctx:stlParser.OpReleaseExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpReleaseExpr"))
        phi1 = self.visit(ctx.getRuleContext().getChild(0))
        if ctx.getRuleContext().getChildCount() == 3:
            phi2 = self.visit(ctx.getRuleContext().getChild(2))
            # TODO: Release without interval?
            #return Not(Until(interval[0], interval[1], Not(phi1), Not(phi2)))
        elif ctx.getRuleContext().getChildCount() == 4: # Optional interval
            phi2 = self.visit(ctx.getRuleContext().getChild(3))
            interval = self.visit(ctx.getRuleContext().getChild(2))
            # TODO: Use Until when implemented in Robustness
            #return Not(Until(interval[0], interval[1], Not(phi1), Not(phi2)))


    # Visit a parse tree produced by stlParser#opNextExpr.
    def visitOpNextExpr(self, ctx:stlParser.OpNextExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpNextExpr"))
        if ctx.getRuleContext().getChildCount() == 2:
            return Next(self.visit(ctx.getRuleContext().getChild(1)))
        elif ctx.getRuleContext().getChildCount() == 3: # Optional interval
            # TODO: Add interval as parameter when supported in Robustness
            return Next(self.visit(ctx.getRuleContext().getChild(2)))


    # Visit a parse tree produced by stlParser#opPropExpr.
    def visitOpPropExpr(self, ctx:stlParser.OpPropExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpPropExpr"))
        phi1 = self.visit(ctx.getRuleContext().getChild(0))
        operator = ctx.getRuleContext().getChild(1).getText()
        phi2 = self.visit(ctx.getRuleContext().getChild(2))
        if operator in ['implies', '->']:
            return Implication(phi1, phi2)
        elif operator in ['iff', '<->']:
            return Equals(phi1, phi2)


    # Visit a parse tree produced by stlParser#opNegExpr.
    def visitOpNegExpr(self, ctx:stlParser.OpNegExprContext):
        print("-----------------------------------{}------------------------------------------".format("OpNegExpr"))
        phi = self.visit(ctx.getRuleContext().getChild(1))
        return Not(phi)


    # Visit a parse tree produced by stlParser#signalParenthesisExpr.
    #TODO: This may be irrelevant as ParenPhiExpr is executed instead of this
    def visitSignalParenthesisExpr(self, ctx:stlParser.SignalParenthesisExprContext):
        print("-----------------------------------{}------------------------------------------".format("SignalParenthesisExpr"))
        return self.visit(ctx.getRuleContext().getChild(1))


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