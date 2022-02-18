#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib

def loadObjective(name):
    if "." in name:
        modulename, classname = name.split(".")
        modulename += "."
    else:
        modulename = ""
        classname = name
    module = importlib.import_module("." + modulename + "objective", "objective")
    try:
        the_class = getattr(module, classname)
    except AttributeError:
        raise Exception("The objective module does not have class '{}'.".format(classname))

    return the_class

def loadObjectiveSelector(name):
    module = importlib.import_module(".objective_selector", "objective")
    try:
        the_class = getattr(module, name)
    except AttributeError:
        raise Exception("The objective_selector module does not have class '{}'.".format(name))

    return the_class

