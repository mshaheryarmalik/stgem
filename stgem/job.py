import os, datetime, logging, random
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
    def __init__(self, description, test_repository, success):
        self.timestamp = datetime.datetime.now()
        self.description = description
        self.test_repository = test_repository
        self.success = success
        self.algorithm_performance = None
        self.model_performance = None

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
        temp_file_name = "{}.tmp".format(file_name)
        with open(temp_file_name, "wb") as file:
            pickle.dump(self,file)
        # then we rename it to its final name
        os.replace( temp_file_name, file_name)

def fix_legacy_job_description(d):
    """
    Convert a job without a step (only algorithm is described) to a job with a single test.
    """

    if "steps" not in d and "algorithm" not in d:
        raise Exception("Job description must have a 'steps' or an 'algorithm' entry.")

    new_step = {}
    new_step["algorithm"] = d["algorithm"]
    del d["algorithm"]

    if "algorithm_parameters" in d:
        new_step["algorithm_parameters"] = d["algorithm_parameters"]
        del d["algorithm_parameters"]

    if "job_parameters" in d:
        new_step["step_parameters"] = d["job_parameters"]

    d["step_1"] = new_step
    d["steps"] = ["step_1"]

    print("The provided job description uses a legacy format. Consider updating it to the new multi step format.")
    try:
        new_description = json.dumps(d, indent=3)
        print(new_description)
    except TypeError:
        print("The job description contains references to python functions. It cannot be represented as JSON.")

    return d

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
                try:
                    current = current[k]
                except KeyError as E:
                    raise Exception("Error accessing key '{}' when copying job description values.".format(E.args[0]))
            return current

        def dict_set(d, s, v):
            pcs = s.split(".")
            current = d
            for k in pcs[:-1]:
                current = current[k]
            current[pcs[-1]] = v

        def perform_copy(d, keys):
            keys = [k for k in keys if k in d]
            for key in keys:
                item = dict_access(d, key)
                if isinstance(item, dict):
                    keys += [key + "." + k for k in item.keys()]
                elif isinstance(item, str) and item.startswith("copy:"):
                    dict_set(d, key, dict_access(d, item[5:]))

        # Convert to multi-step format if needed.
        if not "steps" in self.description:
            self.description = fix_legacy_job_description(self.description)

        # Perform initial copying of values for SUT and its parameters.
        copy_keys = ["sut", "sut_parameters"]
        perform_copy(self.description, copy_keys)

        # Fill in empty values for certain parameters if missing.
        for name in ["sut_parameters", "objective_selector_parameters", "algorithm_parameters"]:
            self.description[name] = self.description.get(name, {})
        if "objective_func_parameters" not in self.description:
            self.description["objective_func_parameters"] = []
        for i in range(len(self.description["objective_func"]) - len(self.description["objective_func_parameters"])):
            self.description["objective_func_parameters"].append({})
        self.description["job_parameters"] = self.description.get("job_parameters", {})
        if "module_path" not in self.description["job_parameters"]:
            self.description["job_parameters"]["module_path"] = None

        # Setup seed.
        # ---------------------------------------------------------------------
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
        # ---------------------------------------------------------------------
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Setup loggers.
        # ---------------------------------------------------------------------
        logger_names = ["algorithm", "model"]
        logging.basicConfig(level=logging.DEBUG, format="%(name)s: %(message)s")
        loggers = {x: logging.getLogger(x) for x in ["algorithm", "model"]}
        for logger in loggers.values():
            logger.setLevel("DEBUG")
        logger = namedtuple("Logger", logger_names)(**loggers)

        # Setup the system under test.
        # ---------------------------------------------------------------------
        sut_class = load_stgem_class(self.description["sut"], "sut", self.description["job_parameters"]["module_path"])
        asut = sut_class(parameters=self.description.get("sut_parameters", {}))
        # Setup input and output names and dimensions if necessary. The values
        # in the job description take precendence.
        if "inputs" in asut.parameters:
            asut.inputs = asut.parameters["inputs"]
        if "input_range" in asut.parameters:
            asut.input_range = asut.parameters["input_range"]
        if hasattr(asut, "idim"):
            # idim set by SUT, check if input names are also set.
            if not hasattr(asut, "inputs"):
                # Set input names to default.
                asut.inputs = ["i{}".format(i) for i in range(asut.idim)]
        else:
            # idim not set by SUT, so we need to figure it out from parameters.
            if hasattr(asut, "inputs"):
                # Input names defined, so infer from it.
                asut.idim = len(asut.inputs)
            else:
                # Input names not defined. The only option is to infer from
                # input ranges. Otherwise we do not know what to do.
                if not hasattr(asut, "input_range"):
                    raise Exception("SUT input dimension not defined and cannot be inferred.")
                asut.idim = len(asut.input_range)
                asut.inputs = ["i{}".format(i) for i in range(asut.idim)]

        if "outputs" in asut.parameters:
            asut.outputs = asut.parameters["outputs"]
        if "output_range" in asut.parameters:
            asut.output_range = asut.parameters["output_range"]
        if hasattr(asut, "odim"):
            # odim set by SUT, check if output names are also set.
            if not hasattr(asut, "outputs"):
                # Set output names to default.
                asut.outputs = ["o{}".format(i) for i in range(asut.odim)]
        else:
            # odim not set by SUT, so we need to figure it out from parameters.
            if hasattr(asut, "outputs"):
                # Output names defined, so infer from it.
                asut.odim = len(asut.outputs)
            else:
                # Output names not defined. The only option is to infer from
                # output ranges. Otherwise we do not know what to do.
                if not hasattr(asut, "output_range"):
                    raise Exception("SUT output dimension not defined and cannot be inferred.")
                asut.odim = len(asut.output_range)
                asut.outputs = ["o{}".format(i) for i in range(asut.odim)]

        # Setup input and output ranges and fill unspecified input and output
        # ranges with Nones.
        if not hasattr(asut, "input_range"):
            asut.input_range = []
        if not isinstance(asut.input_range, list):
            raise Exception("The input_range attribute of the SUT must be a Python list.")
        asut.input_range += [None for _ in range(asut.idim - len(asut.input_range))]
        if not hasattr(asut, "output_range"):
            asut.output_range = []
        if not isinstance(asut.output_range, list):
            raise Exception("The output_range attribute of the SUT must be a Python list.")
        asut.output_range += [None for _ in range(asut.odim - len(asut.output_range))]
        # Run secondary initializer.
        asut.initialize()

        # Copy SUT input dimension to algorithm input_dimension unless it is
        # already defined.
        for step in self.description["steps"]:
            self.description[step]["algorithm_parameters"] = self.description[step].get("algorithm_parameters", {})
            if not "input_dimension" in self.description[step]["algorithm_parameters"]:
                self.description[step]["algorithm_parameters"]["input_dimension"] = asut.idim

        # Setup the test repository.
        # ---------------------------------------------------------------------
        self.test_repository = TestRepository()

        # Setup the objective functions for optimization.
        # ---------------------------------------------------------------------
        # Perform the next copying.
        copy_keys += ["objective_func", "objective_func_parameters"]
        perform_copy(self.description, copy_keys)

        N_objectives = 0
        objective_funcs = []
        for n, s in enumerate(self.description["objective_func"]):
            objective_class = load_stgem_class(s, "objective", self.description["job_parameters"]["module_path"])
            objective_func = objective_class(sut=asut, **self.description["objective_func_parameters"][n])
            N_objectives += objective_func.dim
            objective_funcs.append(objective_func)

        # Setup the objective selector.
        # ---------------------------------------------------------------------
        copy_keys += ["objective_selector", "objective_selector_parameters"]
        perform_copy(self.description, copy_keys)

        objective_selector_class = load_stgem_class(self.description["objective_selector"], "objective_selector", self.description["job_parameters"]["module_path"])
        objective_selector = objective_selector_class(N_objectives=N_objectives, **self.description["objective_selector_parameters"])

        # Create algorithm objects for each step.
        # ---------------------------------------------------------------------
        copy_keys += [step for step in self.description["steps"]]
        perform_copy(self.description, copy_keys)

        self.algorithms = []
        for step_id in self.description["steps"]:
            try:
                step = self.description[step_id]
            except KeyError:
                raise Exception("No step with id '{}'.".format(step_id))

            step["algorithm_parameters"] = step.get("algorithm_parameters", {})

            # Setup the device.
            step["algorithm_parameters"]["device"] = device
            
            # Setup the algorithm.
            algorithm_class = load_stgem_class(step["algorithm"],
                                               "algorithm",
                                               self.description["job_parameters"]["module_path"])

            algorithm = algorithm_class(sut=asut,
                                        test_repository=self.test_repository,
                                        objective_funcs=objective_funcs,
                                        objective_selector=objective_selector,
                                        parameters=step["algorithm_parameters"],
                                        logger=logger)

            self.algorithms.append(algorithm)

        return self

    def run(self) -> JobResult:

        success = False
        # save step results on loop to a list
        step_result_obj_list = []

        # loop through steps and algorithms at the same time and perform steps
        for step_id, algorithm in zip(self.description["steps"], self.algorithms):
            step = self.description[step_id]
            mode = "exhaust_budget" if "mode" not in step["step_parameters"] else step["step_parameters"]["mode"]
            if mode not in ["exhaust_budget", "stop_at_first_objective"]:
                raise Exception("Unknown test generation mode '{}'.".format(mode))

            max_time = step["step_parameters"].get("max_time", 0)
            max_tests = step["step_parameters"].get("max_tests", 0)
            if max_time == 0 and max_tests == 0:
                raise Exception("Step description does not specify neither a maximum time nor a maximum number tests.")

            generator = algorithm.generate_test()
            outputs = []

            i = 0
            start_time = time.perf_counter()
            elapsed_time = 0

            while (max_tests == 0 or i < max_tests) and (max_time == 0 or elapsed_time < max_time):
                try:
                    idx = next(generator)
                except StopIteration:
                    print("Generator finished before budget was exhausted.")
                    break
                _, output = algorithm.test_repository.get(idx)
                outputs.append(output)

                if not success and np.min(output) == 0:
                    print("First success at test {}.".format(i + 1))
                    success = True

                if success and mode == "stop_at_first_objective":
                    break

                i += 1
                elapsed_time = time.perf_counter() - start_time

            if len(outputs) > 0:
                print("Step {}, minimum objective components:".format(step_id))
                print(np.min(np.asarray(outputs), axis=0))

            step_result = StepResult(step, algorithm.test_repository, success)
            step_result.algorithm_performance = algorithm.perf
            step_result.model_performance = [algorithm.models[i].perf for i in range(algorithm.N_models)]

            step_result_obj_list.append(step_result)

            if not success:
                print("Could not fulfill objective within the given budget.")

            if success and mode == "stop_at_first_objective":
                break

        jr = JobResult(self.description, self.test_repository, step_result_obj_list)

        if len(step_result_obj_list) > 0:
            jr.sut_performance = algorithm.sut.perf

        return jr

