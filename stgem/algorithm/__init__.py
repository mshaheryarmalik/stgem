from stgem.algorithm.algorithm import Algorithm
from stgem.algorithm.model import Model, ModelSkeleton

import inspect

def filter_arguments(dictionary, target):
    allowed_keys = [param.name for param in inspect.signature(target).parameters.values() if param.kind == param.POSITIONAL_OR_KEYWORD]
    return {key: dictionary[key] for key in dictionary if key in allowed_keys}

