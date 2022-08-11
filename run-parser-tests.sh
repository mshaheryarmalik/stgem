#!/bin/sh

export PYTHONPATH=`pwd`

cd parser
cd grammar
antlr4 -o .. -visitor -Dlanguage=Python3 stlLexer.g4
antlr4 -o .. -visitor -Dlanguage=Python3 stlParser.g4
cd ..
cd ..

cd tests-parser
pytest