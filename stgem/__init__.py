import importlib

def load_stgem_class(name, namespace, module_path=None):
    # If there is no dot in the name, then we attempt to load the class
    # primarily from the appropriate file in the module path and secondarily
    # under stgem from the appropriate submodule. Otherwise we use the part
    # before the dot to indicate a submodule under stgem and load from there.
    if "." in name:
        pcs = name.split(".")
        primary_module_name = "stgem." + namespace + "." + pcs[0] + "." + namespace
        secondary_module_name = None
        class_name = pcs[1]
    else:
        if module_path is None:
            primary_module_name = "stgem." + namespace + "." + namespace
            secondary_module_name = None
            class_name = name
        else:
            primary_module_name = module_path + "." + namespace
            secondary_module_name = "stgem." + namespace + "." + namespace
            class_name = name

    module = None
    try:
        module_name = primary_module_name
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        if secondary_module_name is not None:
            try:
                module_name = secondary_module_name
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                pass

    if module is None:
        raise Exception("The module '{}' does not exist.".format(module_name))

    try:
        the_class = getattr(module, class_name)
    except:
        raise Exception("The module '{}' does not have class '{}'.".format(module_name, class_name))

    return the_class

