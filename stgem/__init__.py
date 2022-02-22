import importlib

def load_stgem_class(name, namespace, module_path=None):
    # If there is no dot in the name, then we attempt to load the class from
    # the appropriate file from the module path. Otherwise we use the part
    # before the dot to indicate a subfolder under stgem and load from there.
    if "." in name:
        pcs = name.split(".")
        module_name = "stgem." + namespace + "." + pcs[0] + "." + namespace
        class_name = pcs[1]
    else:
        if module_path is None:
            raise Exception("No submodule name specified when importing from stgem namespace {}".format(namespace))
        module_name = module_path + "." + namespace
        class_name = name

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        raise Exception("The module '{}' does not exist.".format(module_name))
    try:
        the_class = getattr(module, class_name)
    except:
        raise Exception("The module '{}' does not have class '{}'.".format(module_name, class_name))

    return the_class

