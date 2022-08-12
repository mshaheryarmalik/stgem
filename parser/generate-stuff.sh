#!/bin/sh
cd grammar
antlr4 -o .. -Dlanguage=Python3 stlLexer.g4 stlParser.g4
#antlr4 -o .. -visitor -Dlanguage=Python3 stlLexer.g4 stlParser.g4
