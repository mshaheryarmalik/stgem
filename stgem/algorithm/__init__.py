#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib, inspect


from stgem import load_stgem_module

def loadAlgorithm(name):
    return load_stgem_module(name,"algorithm")


def filter_arguments(dictionary, target):
    allowed_keys = [param.name for param in inspect.signature(target).parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    return {key: dictionary[key] for key in dictionary if key in allowed_keys}
