
import os, datetime, logging, random
import sys
from collections import namedtuple
import json
import datetime
import numpy as np
import torch
import dill as pickle
import time

from stgem import load_stgem_class
import stgem.algorithm as algorithm
import stgem.objective as objective
from stgem.test_repository import TestRepository

class StepResult:
    def __init__(self, description, test_suite, success):
        self.timestamp = datetime.datetime.now()
        self.description = description
        self.test_suite= test_suite
        self.success = success
        self.algorithm_performance = None
        self.model_performance = None
        self.test_suite = None

class JobResult:
    def __init__(self, description, test_repository, step_results):
        self.timestamp = datetime.datetime.now()
        self.description = description
        self.step_results = step_results
        self.test_repository = test_repository
        self.sut_performance = None

    @staticmethod
    def restore_from_file(file_name):
        with open(file_name, "rb") as file:
            obj=pickle.load(file)
        return obj

    def dump_to_file(self,file_name):
        # first create a temporary file
        temp_file_name=file_name+".tmp"
        with open(temp_file_name,"wb") as file:
            pickle.dump(self,file)
        # then we rename it to its final name
        os.replace( temp_file_name, file_name)

class Job:
    def __init__(self, description=None):
        if description is None:
            self.description = {}
        else:
            self.description = description
            self.setup()

    def setup_from_dict(self, description):
        self.description = description
        self.setup()
        return self

    def setup_from_json(self, json_s):
        d = json.loads(json_s)
        return self.setup_from_dict(d)

    def setup_from_file(self, file_name):
        with open(file_name) as f:
            self.description = json.load(f)
        self.setup()
        return self

    def setup(self):
        def dict_access(d, s):
            current = d
            for k in s.split("."):
                current = current[k]
            return current

        def dict_set(d, s, v):
            pcs = s.split(".")
            current = d
            for k in pcs[:-1]:
                current = current[k]
            current[pcs[-1]] = v

        # Implement the copying mechanism of values.
        keys = list(self.description.keys())
        for key in keys:
            item = dict_access(self.description, key)
            if isinstance(item, dict):
                keys += [key + "." + k for k in item.keys()]
            elif isinstance(item, str) and item.startswith("copy:"):
                dict_set(self.description, key, dict_access(self.description, item[5:]))

        # Fill in empty values for certain parameters if missing.
        for name in ["sut_parameters", "objective_selector_parameters","algorithm_parameters"]:
            if not name in self.description:
                self.description[name] = {}
        if "objective_func_parameters" not in self.description:
            self.description["objective_func_parameters"] = []
        for i in range(len(self.description["objective_func"]) - len(self.description["objective_func_parameters"])):
            self.description["objective_func_parameters"].append({})
        if "module_path" not in self.description["job_parameters"]:
            self.description["job_parameters"]["module_path"] = None

        # Setup seed.
        # We use a random seed unless it is specified.
        # Notice that making Pytorch deterministic makes it a lot slower.
        if "seed" in self.description["job_parameters"] and self.description["job_parameters"]["seed"] is not None:
            SEED = self.description["job_parameters"]["seed"]
            torch.use_deterministic_algorithms(mode=True)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        else:
            SEED = random.randint(0, 2**15)

        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        # Setup loggers.
        logger_names = ["algorithm", "model"]
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
        loggers = {x: logging.getLogger(x) for x in ["algorithm", "model"]}
        for logger in loggers.values():
            logger.setLevel("DEBUG")
        logger = namedtuple("Logger", logger_names)(**loggers)

        # Setup the system under test.
        sut_class = load_stgem_class(self.description["sut"], "sut", self.description["job_parameters"]["module_path"])
        sut_parameters = self.description.get("sut_parameters", {})
        if not "input_range" in sut_parameters:
            sut_parameters["input_range"] = []
        if not "output_range" in sut_parameters:
            sut_parameters["output_range"] = []
        asut = sut_class(parameters=sut_parameters)
        # Setup input and output names and dimensions if necessary.
        if asut.idim is not None:
            if "inputs" not in sut_parameters:
                sut_parameters["inputs"] = ["i{}".format(i) for i in range(asut.idim)]
        else:
            if isinstance(sut_parameters["inputs"], int):
                sut_parameters["inputs"] = ["i{}".format(i) for i in range(asut.idim)]
            asut.idim = len(sut_parameters["inputs"])

        if asut.odim is not None:
            if "outputs" not in sut_parameters:
                sut_parameters["outputs"] = ["o{}".format(i) for i in range(asut.odim)]
        else:
            if isinstance(sut_parameters["outputs"], int):
                sut_parameters["outputs"] = ["o{}".format(i) for i in range(asut.odim)]
            asut.odim = len(sut_parameters["outputs"])

        asut.inputs = sut_parameters["inputs"]
        asut.outputs = sut_parameters["outputs"]

        # Fill in unspecified input and output ranges with Nones.
        asut.irange += [None for _ in range(asut.idim - len(asut.irange))]
        asut.orange += [None for _ in range(asut.odim - len(asut.orange))]
        # Run secondary initializer.
        asut.initialize()

        # Setup the test repository.
        self.test_repository = TestRepository()

        # Setup the objective functions for optimization.
        N_objectives = 0
        objective_funcs = []
        for n, s in enumerate(self.description["objective_func"]):
            objective_class = load_stgem_class(s, "objective", self.description["job_parameters"]["module_path"])
            objective_func = objective_class(sut=asut, **self.description["objective_func_parameters"][n])
            N_objectives += objective_func.dim
            objective_funcs.append(objective_func)

        # Setup the objective selector.
        objective_selector_class = load_stgem_class(self.description["objective_selector"], "objective_selector", self.description["job_parameters"]["module_path"])
        objective_selector = objective_selector_class(N_objectives=N_objectives, **self.description["objective_selector_parameters"])

        # separate steps algorithms as algorithms list object
        algorithm_classes = []
        for step in self.description["steps"]:
            # Setup the device.
            step["algorithm_parameters"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Process job parameters for algorithm setup.
            # Setup the initial random tests to 20% unless the value is user-set.
            if not "N_random_init" in step["step_parameters"]:
                # if max_tests nor N_random_init are provided we use 20 tests
                step["step_parameters"]["N_random_init"] = int(0.2 * step["step_parameters"].get("max_tests", 100))

            # Select the algorithm to be used and setup it.
            # TODO: predefined random data loader
            step["algorithm_parameters"]["max_tests"] = step["step_parameters"].get("max_tests", 0)
            step["algorithm_parameters"]["N_random_init"] = step["step_parameters"]["N_random_init"]

            algorithm_class = load_stgem_class(step["algorithm"],
                                               "algorithm",
                                               self.description["job_parameters"]["module_path"])

            # Select the algorithm to be used and setup it.
            step["algorithm_parameters"]["max_tests"] = step["step_parameters"].get("max_tests", 0)
            step["algorithm_parameters"]["N_random_init"] = step["step_parameters"]["N_random_init"]

            self.algorithm = algorithm_class(sut=asut,
                                             test_repository=self.test_repository,
                                             objective_funcs=objective_funcs,
                                             objective_selector=objective_selector,
                                             parameters=step["algorithm_parameters"],
                                             logger=logger)

            algorithm_classes.append(self.algorithm)

        self.algorithms = algorithm_classes

        return self

    def start(self) -> JobResult:
        # old method name
        return self.run()

    def run(self) -> JobResult:

        success = False
        # save step results on loop to a list
        step_result_obj_list = []

        # loop through steps and algorithms at the same time and perform steps
        for step, algorithm in zip(self.description["steps"], self.algorithms):
            mode = "exhaust_budget" if "mode" not in step["step_parameters"] else step["step_parameters"]["mode"]
            if mode not in ["exhaust_budget", "stop_at_first_objective"]:
                raise Exception("Unknown test generation mode '{}'.".format(mode))

            max_time = step["step_parameters"].get("max_time", 0)
            max_tests = step["step_parameters"].get("max_tests", 0)
            if max_time == 0 and max_tests == 0:
                raise Exception(
                    "Step description does not specify neither a maximum time nor a maximum number tests")

            generator = algorithm.generate_test()
            outputs = []

            i = 0
            start_time = time.perf_counter()
            elapsed_time = 0

            while (max_tests == 0 or i < max_tests) and (max_time == 0 or elapsed_time < max_time):

                idx = next(generator)
                _, output = algorithm.test_repository.get(idx)
                outputs.append(output)

                if not success and np.min(output) == 0:
                    print("First success at test {}.".format(i + 1))
                    success = True

                if success and mode == "stop_at_first_objective":
                    break

                i += 1
                elapsed_time = time.perf_counter() - start_time


            print("Minimum objective components:")
            print(np.min(np.asarray(outputs), axis=0))

            step_result = StepResult(step,algorithm.test_suite, success)
            step_result.algorithm_performance= algorithm.perf
            step_result.model_performance = [ algorithm.models[i].perf for i in range(algorithm.N_models)]

            step_result_obj_list.append(step_result)

            if not success:
                print("Could not fulfill objective within the given budget.")

            if success and mode == "stop_at_first_objective":
                break

        jr = JobResult(self.description, self.test_repository, step_result_obj_list)

        if len(step_result_obj_list)>0:
            jr.sut_performance= algorithm.sut.perf

        return jr
