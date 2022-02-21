import importlib


def load_stgem_module(name,namespace):
    module=None
    pcs = name.split(".")
    try:
        module = importlib.import_module(pcs[0] + "."+namespace)
    except ModuleNotFoundError:
        pass
    if not module:
        try:
            module = importlib.import_module("." + pcs[0] + "."+namespace, package="stgem."+namespace)
        except ModuleNotFoundError:
            raise Exception("The specified {} module '{}' does not exist.".format(namespace, pcs[0]))
    try:
        the_class = getattr(module, pcs[1])
    except AttributeError:
        raise Exception("The specified {} module '{}' does not have class '{}'.".format(namespace, pcs[0], pcs[1]))

    return the_class
