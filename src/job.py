
import os, datetime, logging, random
from collections import namedtuple
import json
import datetime
import numpy as np
import torch
import dill as pickle


import sut, objective, algorithm
from test_repository import TestRepository


class JobResult:
    def __init__(self,descriptipon,test_repository,falsified):
        self.timestamp= datetime.datetime.now()
        self.description=descriptipon
        self.falsified= falsified
        self.test_repository=test_repository
        self.test_suite=None
        self.algorithm_performance=None
        self.model_performance = None
        self.sut_performance=None

    @staticmethod
    def restore_from_file(file_name):
        with open(file_name, "rb") as file:
            obj=pickle.load(file)
        return obj

    def dump_to_file(self,file_name):
        with open(file_name,"wb") as file:
            pickle.dump(self,file)


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
        for i in range(len(self.description["objective_func"]) - len(self.description["objective_func_parameters"])):
            self.description["objective_func_parameters"].append({})

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

        # Setup the device.
        self.description["algorithm_parameters"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup loggers.
        logger_names = ["algorithm", "model"]
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
        loggers = {x: logging.getLogger(x) for x in ["algorithm", "model"]}
        for logger in loggers.values():
            logger.setLevel("DEBUG")
        logger = namedtuple("Logger", logger_names)(**loggers)

        # Setup the system under test.
        sut_class = sut.loadSUT(self.description["sut"])
        asut = sut_class(parameters=self.description.get("sut_parameters",{}))

        # Setup the test repository.
        test_repository = TestRepository()

        # Setup the objective functions for optimization.
        N_objectives = 0
        objective_funcs = []
        for n, s in enumerate(self.description["objective_func"]):
            objective_class = objective.loadObjective(s)
            objective_func = objective_class(sut=asut, **self.description["objective_func_parameters"][n])
            N_objectives += objective_func.dim
            objective_funcs.append(objective_func)

        # Setup the objective selector.
        objective_selector_class = objective.loadObjectiveSelector(self.description["objective_selector"])
        objective_selector = objective_selector_class(N_objectives=N_objectives, **self.description["objective_selector_parameters"])

        # Process job parameters for algorithm setup.
        # Setup the initial random tests to 20% unless the value is user-set.
        if not "N_random_init" in self.description["job_parameters"]:
            self.description["job_parameters"]["N_random_init"] = int(0.2 * self.description["job_parameters"]["N_tests"])

        # Select the algorithm to be used and setup it.
        # TODO: predefined random data loader
        self.description["algorithm_parameters"]["N_tests"] = self.description["job_parameters"]["N_tests"]
        self.description["algorithm_parameters"]["N_random_init"] = self.description["job_parameters"]["N_random_init"]
        algorithm_class = algorithm.loadAlgorithm(self.description["algorithm"])
        self.algorithm = algorithm_class(sut=asut,
                                         test_repository=test_repository,
                                         objective_funcs=objective_funcs,
                                         objective_selector=objective_selector,
                                         parameters=self.description["algorithm_parameters"],
                                         logger=logger
                                        )

        return self

    def start(self) -> JobResult:
        # old method name
        return self.run()

    def run(self) -> JobResult:

        mode = "exhaust_budget" if "mode" not in self.description["job_parameters"] else self.description["job_parameters"]["mode"]
        if mode not in ["exhaust_budget", "stop_at_first_falsification"]:
            raise Exception("Unknown test generation mode '{}'.".format(mode))

        falsified=False

        generator = self.algorithm.generate_test()
        outputs = []

        for i in range(self.description["job_parameters"]["N_tests"]):

            idx = next(generator)
            _, output = self.algorithm.test_repository.get(idx)
            outputs.append(output)

            if not falsified and np.min(output) == 0:
                print("First falsified at test {}.".format(i + 1))
                falsified = True
            if falsified and mode == "stop_at_first_falsification":
                break

        if not falsified:
            print("Could not falsify within the given budget.")

        print("Minimum objective components:")
        print(np.min(np.asarray(outputs), axis=0))

        jr=JobResult(self.description,self.algorithm.test_repository,falsified)
        jr.algorithm_performance=self.algorithm.perf
        jr.sut_performance=self.algorithm.sut.perf
        jr.test_suite = self.algorithm.test_suite
        jr.model_performance = [self.algorithm.models[i].perf for i in range(self.algorithm.N_models)]

        return jr

