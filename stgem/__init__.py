import importlib
import sys
import traceback


def load_stgem_class(name, namespace, module_path=None):
    # If there is no dot in the name, then we attempt to load the class
    # primarily from the appropriate file in the module path and secondarily
    # under stgem from the appropriate submodule. Otherwise we use the part
    # before the dot to indicate a submodule under stgem and load from there.

    if "." in name:
        pcs = name.split(".")
        primary_module_name = "stgem.{}.{}.{}".format(namespace, pcs[0], namespace)
        secondary_module_name = None
        class_name = pcs[1]
    else:
        if module_path is None:
            primary_module_name = "stgem.{}.{}".format(namespace, namespace)
            secondary_module_name = None
            class_name = name
        else:
            primary_module_name = "{}.{}".format(module_path, namespace)
            secondary_module_name = "stgem.{}.{}".format(namespace, namespace)
            class_name = name

    module = None
    traceback_record = str
    try:
        module_name = primary_module_name
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        traceback_record = traceback.format_exc()
        if secondary_module_name is not None:
            try:
                module_name = secondary_module_name
                module = importlib.import_module(module_name)
            except ModuleNotFoundError:
                traceback_record = traceback.format_exc()

    if module is None:
        print(traceback_record)
        print("Error importing module '{}'.".format(module_name))
        raise ModuleNotFoundError

    try:
        the_class = getattr(module, class_name)
    except Exception as err:
        print("Error importing class '{}.{}'.".format(module_name,class_name))
        raise err

    return the_class
