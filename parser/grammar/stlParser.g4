parser grammar stlParser;

options {
    language = Python3;
    tokenVocab = stlLexer;
}

stlSpecification
    : phi EOF ;

phi
    : LPAREN phi RPAREN              #parenPhiExpr

    | NEGATION phi                   #opNegExpr

    | NEXTOP        (interval)? phi  #opNextExpr
    | FUTUREOP      (interval)? phi  #opFutureExpr
    | GLOBALLYOP    (interval)? phi  #opGloballyExpr

    | phi UNTILOP   (interval)? phi  #opUntilExpr
    | phi RELEASEOP (interval)? phi  #opReleaseExpr

    | phi (ANDOP | OROP) phi         #opLogicalExpr

    | phi (IMPLIESOP | EQUIVOP) phi  #opPropExpr

    | signal RELOP signal            #predicateExpr
    | signal                         #signalExpr
;

signal
    : NUMBER                         #signalNumber
    | NAME                           #signalName
    | (LPAREN) signal (RPAREN)       #signalParenthesisExpr
    | (SUM | MINUS) signal           #signalUnaryExpr
    | signal (MULT | DIV) signal     #signalMultExpr
    | signal (SUM | MINUS) signal    #signalSumExpr
;

interval
    : (LPAREN | LBRACK) (NUMBER | INF) COMMA (NUMBER | INF) (RPAREN | RBRACK)
;
