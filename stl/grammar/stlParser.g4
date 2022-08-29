parser grammar stlParser;

options {
    language = Python3;
    tokenVocab = stlLexer;
}

stlSpecification
    : phi EOF ;

phi
    : LPAREN phi RPAREN                  #parenPhiExpr

    | NEGATION phi                       #opNegExpr
    | NEXTOP phi                         #opNextExpr

    | FUTUREOP      (interval)? phi      #opFutureExpr
    | GLOBALLYOP    (interval)? phi      #opGloballyExpr

    | phi UNTILOP   (interval)? phi      #opUntilExpr

    | phi ANDOP phi                      #opAndExpr
    | phi OROP phi                       #opOrExpr

    | phi (IMPLIESOP | EQUIVOP) phi      #opPropExpr

    | signal (RELOP | EQUALITYOP) signal #predicateExpr
    | signal                             #signalExpr
;

signal
    : NUMBER                         #signalNumber
    | NAME                           #signalName
    | LPAREN signal RPAREN           #signalParenthesisExpr
    | signal (MULT | DIV) signal     #signalMultExpr
    | signal (PLUS | MINUS) signal   #signalSumExpr
    | VBAR signal VBAR               #signalAbsExpr
;

interval
    : (LPAREN | LBRACK) (NUMBER | INF) COMMA (NUMBER | INF) (RPAREN | RBRACK)
;
