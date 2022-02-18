#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib

from .sut import SUT

def loadSUT(name):
    pcs = name.split(".")
    try:
        module = importlib.import_module("." + pcs[0] + ".sut", package="stgem.sut")
    except ModuleNotFoundError:
        raise Exception("The specified SUT module '{}' does not exist.".format(pcs[0]))
    try:
        the_class = getattr(module, pcs[1])
    except AttributeError:
        raise Exception("The specified SUT module '{}' does not have class '{}'.".format(pcs[0], pcs[1]))

    return the_class
