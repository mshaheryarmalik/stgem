#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib

from .objective import Objective
from .objective_selector import ObjectiveSelector
from stgem import load_stgem_module

def loadObjective(name):
    return load_stgem_module(name,"objective.objective")

def loadObjectiveSelector(name):
    module = importlib.import_module(".objective_selector", "stgem.objective")
    try:
        the_class = getattr(module, name)
    except AttributeError:
        raise Exception("The objective_selector module does not have class '{}'.".format(name))

    return the_class

