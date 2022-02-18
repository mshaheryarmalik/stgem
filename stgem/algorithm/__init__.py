#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib, inspect

from .algorithm import Algorithm
from .model import Model

def loadAlgorithm(name):
    pcs = name.split(".")
    try:
        module = importlib.import_module("." + pcs[0] + ".algorithm", "stgem.algorithm")
    except ModuleNotFoundError:
        raise Exception("The specified algorithm module '{}' does not exist.".format(pcs[0]))
    try:
        class_algorithm = getattr(module, pcs[1])
    except AttributeError:
        raise Exception("The algorithm module does not have class '{}'.".format(pcs[1]))

    return class_algorithm


def filter_arguments(dictionary, target):
    allowed_keys = [param.name for param in inspect.signature(target).parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    return {key: dictionary[key] for key in dictionary if key in allowed_keys}
