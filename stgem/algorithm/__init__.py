#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib, inspect
from .algorithm import Algorithm
from .model import Model

from stgem import load_stgem_class

def loadAlgorithm(name):
    return load_stgem_class(name, "algorithm")

def filter_arguments(dictionary, target):
    allowed_keys = [param.name for param in inspect.signature(target).parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    return {key: dictionary[key] for key in dictionary if key in allowed_keys}
