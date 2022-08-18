# Generated from stlParser.g4 by ANTLR 4.10.1
# encoding: utf-8
from antlr4 import *
from io import StringIO
import sys
if sys.version_info[1] > 5:
	from typing import TextIO
else:
	from typing.io import TextIO

def serializedATN():
    return [
        4,1,28,90,2,0,7,0,2,1,7,1,2,2,7,2,2,3,7,3,1,0,1,0,1,0,1,1,1,1,1,
        1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,21,8,1,1,1,1,1,1,1,3,1,26,8,1,1,1,
        1,1,1,1,3,1,31,8,1,1,1,1,1,1,1,1,1,1,1,1,1,3,1,39,8,1,1,1,1,1,1,
        1,3,1,44,8,1,1,1,1,1,1,1,1,1,3,1,50,8,1,1,1,1,1,1,1,1,1,1,1,1,1,
        1,1,5,1,59,8,1,10,1,12,1,62,9,1,1,2,1,2,1,2,1,2,1,2,1,2,1,2,3,2,
        71,8,2,1,2,1,2,1,2,1,2,1,2,1,2,5,2,79,8,2,10,2,12,2,82,9,2,1,3,1,
        3,1,3,1,3,1,3,1,3,1,3,0,2,2,4,4,0,2,4,6,0,7,1,0,19,20,1,0,21,22,
        1,0,9,10,1,0,7,8,2,0,2,2,4,4,2,0,23,23,25,25,2,0,3,3,5,5,104,0,8,
        1,0,0,0,2,38,1,0,0,0,4,70,1,0,0,0,6,83,1,0,0,0,8,9,3,2,1,0,9,10,
        5,0,0,1,10,1,1,0,0,0,11,12,6,1,-1,0,12,13,5,2,0,0,13,14,3,2,1,0,
        14,15,5,3,0,0,15,39,1,0,0,0,16,17,5,11,0,0,17,39,3,2,1,10,18,20,
        5,14,0,0,19,21,3,6,3,0,20,19,1,0,0,0,20,21,1,0,0,0,21,22,1,0,0,0,
        22,39,3,2,1,9,23,25,5,15,0,0,24,26,3,6,3,0,25,24,1,0,0,0,25,26,1,
        0,0,0,26,27,1,0,0,0,27,39,3,2,1,8,28,30,5,16,0,0,29,31,3,6,3,0,30,
        29,1,0,0,0,30,31,1,0,0,0,31,32,1,0,0,0,32,39,3,2,1,7,33,34,3,4,2,
        0,34,35,5,12,0,0,35,36,3,4,2,0,36,39,1,0,0,0,37,39,3,4,2,0,38,11,
        1,0,0,0,38,16,1,0,0,0,38,18,1,0,0,0,38,23,1,0,0,0,38,28,1,0,0,0,
        38,33,1,0,0,0,38,37,1,0,0,0,39,60,1,0,0,0,40,41,10,6,0,0,41,43,5,
        17,0,0,42,44,3,6,3,0,43,42,1,0,0,0,43,44,1,0,0,0,44,45,1,0,0,0,45,
        59,3,2,1,7,46,47,10,5,0,0,47,49,5,18,0,0,48,50,3,6,3,0,49,48,1,0,
        0,0,49,50,1,0,0,0,50,51,1,0,0,0,51,59,3,2,1,6,52,53,10,4,0,0,53,
        54,7,0,0,0,54,59,3,2,1,5,55,56,10,3,0,0,56,57,7,1,0,0,57,59,3,2,
        1,4,58,40,1,0,0,0,58,46,1,0,0,0,58,52,1,0,0,0,58,55,1,0,0,0,59,62,
        1,0,0,0,60,58,1,0,0,0,60,61,1,0,0,0,61,3,1,0,0,0,62,60,1,0,0,0,63,
        64,6,2,-1,0,64,71,5,25,0,0,65,71,5,24,0,0,66,67,5,2,0,0,67,68,3,
        4,2,0,68,69,5,3,0,0,69,71,1,0,0,0,70,63,1,0,0,0,70,65,1,0,0,0,70,
        66,1,0,0,0,71,80,1,0,0,0,72,73,10,2,0,0,73,74,7,2,0,0,74,79,3,4,
        2,3,75,76,10,1,0,0,76,77,7,3,0,0,77,79,3,4,2,2,78,72,1,0,0,0,78,
        75,1,0,0,0,79,82,1,0,0,0,80,78,1,0,0,0,80,81,1,0,0,0,81,5,1,0,0,
        0,82,80,1,0,0,0,83,84,7,4,0,0,84,85,7,5,0,0,85,86,5,6,0,0,86,87,
        7,5,0,0,87,88,7,6,0,0,88,7,1,0,0,0,11,20,25,30,38,43,49,58,60,70,
        78,80
    ]

class stlParser ( Parser ):

    grammarFileName = "stlParser.g4"

    atn = ATNDeserializer().deserialize(serializedATN())

    decisionsToDFA = [ DFA(ds, i) for i, ds in enumerate(atn.decisionToState) ]

    sharedContextCache = PredictionContextCache()

    literalNames = [ "<INVALID>", "<INVALID>", "'('", "')'", "'['", "']'", 
                     "','", "'+'", "'-'", "'*'", "'/'", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "<INVALID>", "<INVALID>", 
                     "<INVALID>", "<INVALID>", "'inf'" ]

    symbolicNames = [ "<INVALID>", "WS", "LPAREN", "RPAREN", "LBRACK", "RBRACK", 
                      "COMMA", "PLUS", "MINUS", "MULT", "DIV", "NEGATION", 
                      "RELOP", "EQUALITYOP", "NEXTOP", "FUTUREOP", "GLOBALLYOP", 
                      "UNTILOP", "RELEASEOP", "ANDOP", "OROP", "IMPLIESOP", 
                      "EQUIVOP", "INF", "NAME", "NUMBER", "INT_NUMBER", 
                      "FLOAT_NUMBER", "SCIENTIFIC_NUMBER" ]

    RULE_stlSpecification = 0
    RULE_phi = 1
    RULE_signal = 2
    RULE_interval = 3

    ruleNames =  [ "stlSpecification", "phi", "signal", "interval" ]

    EOF = Token.EOF
    WS=1
    LPAREN=2
    RPAREN=3
    LBRACK=4
    RBRACK=5
    COMMA=6
    PLUS=7
    MINUS=8
    MULT=9
    DIV=10
    NEGATION=11
    RELOP=12
    EQUALITYOP=13
    NEXTOP=14
    FUTUREOP=15
    GLOBALLYOP=16
    UNTILOP=17
    RELEASEOP=18
    ANDOP=19
    OROP=20
    IMPLIESOP=21
    EQUIVOP=22
    INF=23
    NAME=24
    NUMBER=25
    INT_NUMBER=26
    FLOAT_NUMBER=27
    SCIENTIFIC_NUMBER=28

    def __init__(self, input:TokenStream, output:TextIO = sys.stdout):
        super().__init__(input, output)
        self.checkVersion("4.10.1")
        self._interp = ParserATNSimulator(self, self.atn, self.decisionsToDFA, self.sharedContextCache)
        self._predicates = None




    class StlSpecificationContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def phi(self):
            return self.getTypedRuleContext(stlParser.PhiContext,0)


        def EOF(self):
            return self.getToken(stlParser.EOF, 0)

        def getRuleIndex(self):
            return stlParser.RULE_stlSpecification

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitStlSpecification" ):
                return visitor.visitStlSpecification(self)
            else:
                return visitor.visitChildren(self)




    def stlSpecification(self):

        localctx = stlParser.StlSpecificationContext(self, self._ctx, self.state)
        self.enterRule(localctx, 0, self.RULE_stlSpecification)
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 8
            self.phi(0)
            self.state = 9
            self.match(stlParser.EOF)
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx


    class PhiContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return stlParser.RULE_phi

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)


    class PredicateExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def signal(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(stlParser.SignalContext)
            else:
                return self.getTypedRuleContext(stlParser.SignalContext,i)

        def RELOP(self):
            return self.getToken(stlParser.RELOP, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitPredicateExpr" ):
                return visitor.visitPredicateExpr(self)
            else:
                return visitor.visitChildren(self)


    class SignalExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def signal(self):
            return self.getTypedRuleContext(stlParser.SignalContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSignalExpr" ):
                return visitor.visitSignalExpr(self)
            else:
                return visitor.visitChildren(self)


    class OpFutureExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def FUTUREOP(self):
            return self.getToken(stlParser.FUTUREOP, 0)
        def phi(self):
            return self.getTypedRuleContext(stlParser.PhiContext,0)

        def interval(self):
            return self.getTypedRuleContext(stlParser.IntervalContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOpFutureExpr" ):
                return visitor.visitOpFutureExpr(self)
            else:
                return visitor.visitChildren(self)


    class ParenPhiExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def LPAREN(self):
            return self.getToken(stlParser.LPAREN, 0)
        def phi(self):
            return self.getTypedRuleContext(stlParser.PhiContext,0)

        def RPAREN(self):
            return self.getToken(stlParser.RPAREN, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitParenPhiExpr" ):
                return visitor.visitParenPhiExpr(self)
            else:
                return visitor.visitChildren(self)


    class OpUntilExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def phi(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(stlParser.PhiContext)
            else:
                return self.getTypedRuleContext(stlParser.PhiContext,i)

        def UNTILOP(self):
            return self.getToken(stlParser.UNTILOP, 0)
        def interval(self):
            return self.getTypedRuleContext(stlParser.IntervalContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOpUntilExpr" ):
                return visitor.visitOpUntilExpr(self)
            else:
                return visitor.visitChildren(self)


    class OpGloballyExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def GLOBALLYOP(self):
            return self.getToken(stlParser.GLOBALLYOP, 0)
        def phi(self):
            return self.getTypedRuleContext(stlParser.PhiContext,0)

        def interval(self):
            return self.getTypedRuleContext(stlParser.IntervalContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOpGloballyExpr" ):
                return visitor.visitOpGloballyExpr(self)
            else:
                return visitor.visitChildren(self)


    class OpLogicalExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def phi(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(stlParser.PhiContext)
            else:
                return self.getTypedRuleContext(stlParser.PhiContext,i)

        def ANDOP(self):
            return self.getToken(stlParser.ANDOP, 0)
        def OROP(self):
            return self.getToken(stlParser.OROP, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOpLogicalExpr" ):
                return visitor.visitOpLogicalExpr(self)
            else:
                return visitor.visitChildren(self)


    class OpReleaseExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def phi(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(stlParser.PhiContext)
            else:
                return self.getTypedRuleContext(stlParser.PhiContext,i)

        def RELEASEOP(self):
            return self.getToken(stlParser.RELEASEOP, 0)
        def interval(self):
            return self.getTypedRuleContext(stlParser.IntervalContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOpReleaseExpr" ):
                return visitor.visitOpReleaseExpr(self)
            else:
                return visitor.visitChildren(self)


    class OpNextExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def NEXTOP(self):
            return self.getToken(stlParser.NEXTOP, 0)
        def phi(self):
            return self.getTypedRuleContext(stlParser.PhiContext,0)

        def interval(self):
            return self.getTypedRuleContext(stlParser.IntervalContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOpNextExpr" ):
                return visitor.visitOpNextExpr(self)
            else:
                return visitor.visitChildren(self)


    class OpPropExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def phi(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(stlParser.PhiContext)
            else:
                return self.getTypedRuleContext(stlParser.PhiContext,i)

        def IMPLIESOP(self):
            return self.getToken(stlParser.IMPLIESOP, 0)
        def EQUIVOP(self):
            return self.getToken(stlParser.EQUIVOP, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOpPropExpr" ):
                return visitor.visitOpPropExpr(self)
            else:
                return visitor.visitChildren(self)


    class OpNegExprContext(PhiContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.PhiContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def NEGATION(self):
            return self.getToken(stlParser.NEGATION, 0)
        def phi(self):
            return self.getTypedRuleContext(stlParser.PhiContext,0)


        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitOpNegExpr" ):
                return visitor.visitOpNegExpr(self)
            else:
                return visitor.visitChildren(self)



    def phi(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = stlParser.PhiContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 2
        self.enterRecursionRule(localctx, 2, self.RULE_phi, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 38
            self._errHandler.sync(self)
            la_ = self._interp.adaptivePredict(self._input,3,self._ctx)
            if la_ == 1:
                localctx = stlParser.ParenPhiExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 12
                self.match(stlParser.LPAREN)
                self.state = 13
                self.phi(0)
                self.state = 14
                self.match(stlParser.RPAREN)
                pass

            elif la_ == 2:
                localctx = stlParser.OpNegExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 16
                self.match(stlParser.NEGATION)
                self.state = 17
                self.phi(10)
                pass

            elif la_ == 3:
                localctx = stlParser.OpNextExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 18
                self.match(stlParser.NEXTOP)
                self.state = 20
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,0,self._ctx)
                if la_ == 1:
                    self.state = 19
                    self.interval()


                self.state = 22
                self.phi(9)
                pass

            elif la_ == 4:
                localctx = stlParser.OpFutureExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 23
                self.match(stlParser.FUTUREOP)
                self.state = 25
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,1,self._ctx)
                if la_ == 1:
                    self.state = 24
                    self.interval()


                self.state = 27
                self.phi(8)
                pass

            elif la_ == 5:
                localctx = stlParser.OpGloballyExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 28
                self.match(stlParser.GLOBALLYOP)
                self.state = 30
                self._errHandler.sync(self)
                la_ = self._interp.adaptivePredict(self._input,2,self._ctx)
                if la_ == 1:
                    self.state = 29
                    self.interval()


                self.state = 32
                self.phi(7)
                pass

            elif la_ == 6:
                localctx = stlParser.PredicateExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 33
                self.signal(0)
                self.state = 34
                self.match(stlParser.RELOP)
                self.state = 35
                self.signal(0)
                pass

            elif la_ == 7:
                localctx = stlParser.SignalExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 37
                self.signal(0)
                pass


            self._ctx.stop = self._input.LT(-1)
            self.state = 60
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,7,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 58
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,6,self._ctx)
                    if la_ == 1:
                        localctx = stlParser.OpUntilExprContext(self, stlParser.PhiContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_phi)
                        self.state = 40
                        if not self.precpred(self._ctx, 6):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 6)")
                        self.state = 41
                        self.match(stlParser.UNTILOP)
                        self.state = 43
                        self._errHandler.sync(self)
                        la_ = self._interp.adaptivePredict(self._input,4,self._ctx)
                        if la_ == 1:
                            self.state = 42
                            self.interval()


                        self.state = 45
                        self.phi(7)
                        pass

                    elif la_ == 2:
                        localctx = stlParser.OpReleaseExprContext(self, stlParser.PhiContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_phi)
                        self.state = 46
                        if not self.precpred(self._ctx, 5):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 5)")
                        self.state = 47
                        self.match(stlParser.RELEASEOP)
                        self.state = 49
                        self._errHandler.sync(self)
                        la_ = self._interp.adaptivePredict(self._input,5,self._ctx)
                        if la_ == 1:
                            self.state = 48
                            self.interval()


                        self.state = 51
                        self.phi(6)
                        pass

                    elif la_ == 3:
                        localctx = stlParser.OpLogicalExprContext(self, stlParser.PhiContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_phi)
                        self.state = 52
                        if not self.precpred(self._ctx, 4):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 4)")
                        self.state = 53
                        _la = self._input.LA(1)
                        if not(_la==stlParser.ANDOP or _la==stlParser.OROP):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 54
                        self.phi(5)
                        pass

                    elif la_ == 4:
                        localctx = stlParser.OpPropExprContext(self, stlParser.PhiContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_phi)
                        self.state = 55
                        if not self.precpred(self._ctx, 3):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 3)")
                        self.state = 56
                        _la = self._input.LA(1)
                        if not(_la==stlParser.IMPLIESOP or _la==stlParser.EQUIVOP):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 57
                        self.phi(4)
                        pass

             
                self.state = 62
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,7,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class SignalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser


        def getRuleIndex(self):
            return stlParser.RULE_signal

     
        def copyFrom(self, ctx:ParserRuleContext):
            super().copyFrom(ctx)


    class SignalParenthesisExprContext(SignalContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.SignalContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def signal(self):
            return self.getTypedRuleContext(stlParser.SignalContext,0)

        def LPAREN(self):
            return self.getToken(stlParser.LPAREN, 0)
        def RPAREN(self):
            return self.getToken(stlParser.RPAREN, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSignalParenthesisExpr" ):
                return visitor.visitSignalParenthesisExpr(self)
            else:
                return visitor.visitChildren(self)


    class SignalNameContext(SignalContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.SignalContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def NAME(self):
            return self.getToken(stlParser.NAME, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSignalName" ):
                return visitor.visitSignalName(self)
            else:
                return visitor.visitChildren(self)


    class SignalSumExprContext(SignalContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.SignalContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def signal(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(stlParser.SignalContext)
            else:
                return self.getTypedRuleContext(stlParser.SignalContext,i)

        def PLUS(self):
            return self.getToken(stlParser.PLUS, 0)
        def MINUS(self):
            return self.getToken(stlParser.MINUS, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSignalSumExpr" ):
                return visitor.visitSignalSumExpr(self)
            else:
                return visitor.visitChildren(self)


    class SignalNumberContext(SignalContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.SignalContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def NUMBER(self):
            return self.getToken(stlParser.NUMBER, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSignalNumber" ):
                return visitor.visitSignalNumber(self)
            else:
                return visitor.visitChildren(self)


    class SignalMultExprContext(SignalContext):

        def __init__(self, parser, ctx:ParserRuleContext): # actually a stlParser.SignalContext
            super().__init__(parser)
            self.copyFrom(ctx)

        def signal(self, i:int=None):
            if i is None:
                return self.getTypedRuleContexts(stlParser.SignalContext)
            else:
                return self.getTypedRuleContext(stlParser.SignalContext,i)

        def MULT(self):
            return self.getToken(stlParser.MULT, 0)
        def DIV(self):
            return self.getToken(stlParser.DIV, 0)

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitSignalMultExpr" ):
                return visitor.visitSignalMultExpr(self)
            else:
                return visitor.visitChildren(self)



    def signal(self, _p:int=0):
        _parentctx = self._ctx
        _parentState = self.state
        localctx = stlParser.SignalContext(self, self._ctx, _parentState)
        _prevctx = localctx
        _startState = 4
        self.enterRecursionRule(localctx, 4, self.RULE_signal, _p)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 70
            self._errHandler.sync(self)
            token = self._input.LA(1)
            if token in [stlParser.NUMBER]:
                localctx = stlParser.SignalNumberContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 64
                self.match(stlParser.NUMBER)
                pass
            elif token in [stlParser.NAME]:
                localctx = stlParser.SignalNameContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx
                self.state = 65
                self.match(stlParser.NAME)
                pass
            elif token in [stlParser.LPAREN]:
                localctx = stlParser.SignalParenthesisExprContext(self, localctx)
                self._ctx = localctx
                _prevctx = localctx

                self.state = 66
                self.match(stlParser.LPAREN)
                self.state = 67
                self.signal(0)

                self.state = 68
                self.match(stlParser.RPAREN)
                pass
            else:
                raise NoViableAltException(self)

            self._ctx.stop = self._input.LT(-1)
            self.state = 80
            self._errHandler.sync(self)
            _alt = self._interp.adaptivePredict(self._input,10,self._ctx)
            while _alt!=2 and _alt!=ATN.INVALID_ALT_NUMBER:
                if _alt==1:
                    if self._parseListeners is not None:
                        self.triggerExitRuleEvent()
                    _prevctx = localctx
                    self.state = 78
                    self._errHandler.sync(self)
                    la_ = self._interp.adaptivePredict(self._input,9,self._ctx)
                    if la_ == 1:
                        localctx = stlParser.SignalMultExprContext(self, stlParser.SignalContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_signal)
                        self.state = 72
                        if not self.precpred(self._ctx, 2):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 2)")
                        self.state = 73
                        _la = self._input.LA(1)
                        if not(_la==stlParser.MULT or _la==stlParser.DIV):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 74
                        self.signal(3)
                        pass

                    elif la_ == 2:
                        localctx = stlParser.SignalSumExprContext(self, stlParser.SignalContext(self, _parentctx, _parentState))
                        self.pushNewRecursionContext(localctx, _startState, self.RULE_signal)
                        self.state = 75
                        if not self.precpred(self._ctx, 1):
                            from antlr4.error.Errors import FailedPredicateException
                            raise FailedPredicateException(self, "self.precpred(self._ctx, 1)")
                        self.state = 76
                        _la = self._input.LA(1)
                        if not(_la==stlParser.PLUS or _la==stlParser.MINUS):
                            self._errHandler.recoverInline(self)
                        else:
                            self._errHandler.reportMatch(self)
                            self.consume()
                        self.state = 77
                        self.signal(2)
                        pass

             
                self.state = 82
                self._errHandler.sync(self)
                _alt = self._interp.adaptivePredict(self._input,10,self._ctx)

        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.unrollRecursionContexts(_parentctx)
        return localctx


    class IntervalContext(ParserRuleContext):
        __slots__ = 'parser'

        def __init__(self, parser, parent:ParserRuleContext=None, invokingState:int=-1):
            super().__init__(parent, invokingState)
            self.parser = parser

        def COMMA(self):
            return self.getToken(stlParser.COMMA, 0)

        def LPAREN(self):
            return self.getToken(stlParser.LPAREN, 0)

        def LBRACK(self):
            return self.getToken(stlParser.LBRACK, 0)

        def NUMBER(self, i:int=None):
            if i is None:
                return self.getTokens(stlParser.NUMBER)
            else:
                return self.getToken(stlParser.NUMBER, i)

        def INF(self, i:int=None):
            if i is None:
                return self.getTokens(stlParser.INF)
            else:
                return self.getToken(stlParser.INF, i)

        def RPAREN(self):
            return self.getToken(stlParser.RPAREN, 0)

        def RBRACK(self):
            return self.getToken(stlParser.RBRACK, 0)

        def getRuleIndex(self):
            return stlParser.RULE_interval

        def accept(self, visitor:ParseTreeVisitor):
            if hasattr( visitor, "visitInterval" ):
                return visitor.visitInterval(self)
            else:
                return visitor.visitChildren(self)




    def interval(self):

        localctx = stlParser.IntervalContext(self, self._ctx, self.state)
        self.enterRule(localctx, 6, self.RULE_interval)
        self._la = 0 # Token type
        try:
            self.enterOuterAlt(localctx, 1)
            self.state = 83
            _la = self._input.LA(1)
            if not(_la==stlParser.LPAREN or _la==stlParser.LBRACK):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 84
            _la = self._input.LA(1)
            if not(_la==stlParser.INF or _la==stlParser.NUMBER):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 85
            self.match(stlParser.COMMA)
            self.state = 86
            _la = self._input.LA(1)
            if not(_la==stlParser.INF or _la==stlParser.NUMBER):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
            self.state = 87
            _la = self._input.LA(1)
            if not(_la==stlParser.RPAREN or _la==stlParser.RBRACK):
                self._errHandler.recoverInline(self)
            else:
                self._errHandler.reportMatch(self)
                self.consume()
        except RecognitionException as re:
            localctx.exception = re
            self._errHandler.reportError(self, re)
            self._errHandler.recover(self, re)
        finally:
            self.exitRule()
        return localctx



    def sempred(self, localctx:RuleContext, ruleIndex:int, predIndex:int):
        if self._predicates == None:
            self._predicates = dict()
        self._predicates[1] = self.phi_sempred
        self._predicates[2] = self.signal_sempred
        pred = self._predicates.get(ruleIndex, None)
        if pred is None:
            raise Exception("No predicate with index:" + str(ruleIndex))
        else:
            return pred(localctx, predIndex)

    def phi_sempred(self, localctx:PhiContext, predIndex:int):
            if predIndex == 0:
                return self.precpred(self._ctx, 6)
         

            if predIndex == 1:
                return self.precpred(self._ctx, 5)
         

            if predIndex == 2:
                return self.precpred(self._ctx, 4)
         

            if predIndex == 3:
                return self.precpred(self._ctx, 3)
         

    def signal_sempred(self, localctx:SignalContext, predIndex:int):
            if predIndex == 4:
                return self.precpred(self._ctx, 2)
         

            if predIndex == 5:
                return self.precpred(self._ctx, 1)
         




