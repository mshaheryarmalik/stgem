#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os, datetime, logging
from collections import namedtuple

import torch

import sut, objective, algorithm
from test_repository import TestRepository

class OwnExperiment:

  def __init__(self, algorithm):
    self.algorithm = algorithm

  def start(self):
    generator = self.algorithm.generate_test()
    for i in range(300):
      next(generator)

if __name__ == "__main__":
  # Random
  job = {"sut": "sbst.SBSTSUT",
         "sut_parameters": {"beamng_home": "C:\\Users\\japeltom\\BeamNG\\BeamNG.tech.v0.24.0.1", "map_size": 200, "max_speed": 75, "curvature_points": 5},
         "objective_func": "ObjectiveMaxComponentwise",
         "objective_selector": "ObjectiveSelectorMAB",
         "objective_selector_parameters": {"warm_up": 3},
         "algorithm": "random.Random",
         "algorithm_parameters": {"use_predefined_random_data": False,
                                  "predefined_random_data": {"test_inputs": None,
                                                             "test_outputs": None}},
         "job_parameters": {"N_tests": 10}
         }
  """
  job = {"sut": "odroid.OdroidSUT",
         "sut_parameters": {"data_file": "..\data\odroid\odroid.npy"},
         "objective_func": "ObjectiveMaxSelected",
         "objective_selector": "ObjectiveSelectorMAB",
         "objective_selector_parameters": {"warm_up": 3},
         "algorithm": "random.Random",
         "algorithm_parameters": {"use_predefined_random_data": False,
                                  "predefined_random_data": {"test_inputs": None,
                                                             "test_outputs": None}},
         "job_parameters": {"N_tests": 10}
         }
  """

  # Process the settings copy commands.
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
  keys = list(job.keys())
  for key in keys:
    item = dict_access(job, key)
    if isinstance(item, dict):
      keys += [key + "." + k for k in item.keys()]
    elif isinstance(item, str) and item.startswith("copy:"):
      dict_set(job, key, dict_access(job, item[5:]))

  # Setup the device.
  job["algorithm_parameters"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  # Setup loggers.
  logger_names = ["algorithm", "model"]
  logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")
  loggers = {x:logging.getLogger(x) for x in ["algorithm", "model"]}
  for logger in loggers.values():
    logger.setLevel("INFO")
  logger = namedtuple("Logger", logger_names)(**loggers)

  # Setup the system under test.
  sut_class = sut.loadSUT(job["sut"])
  if not "sut_parameters" in job:
      job["sut_parameters"] = {}
  sut = sut_class(**job["sut_parameters"])

  # Setup the test repository.
  test_repository = TestRepository()

  # Setup the objective functions for optimization.
  objective_class = objective.loadObjective(job["objective_func"])
  if not "objective_func_parameters" in job:
      job["objective_func_parameters"] = {}
  objective_func = objective_class(**job["objective_func_parameters"])
  target = None

  # Setup the objective selector.
  objective_class = objective.loadObjectiveSelector(job["objective_selector"])
  if not "objective_selector_parameters" in job:
    job["objective_selector_parameters"] = {}
  objective_selector = objective_class(objective_func=objective_func, **job["objective_selector_parameters"])

  # Process job parameters for algorithm setup.
  # Setup the initial random tests to 20% unless the value is user-set.
  if not "N_random_init" in job["job_parameters"]:
    job["job_parameters"]["N_random_init"] = int(0.2*job["job_parameters"]["N_tests"])

  # Select the algorithm to be used and setup it.
  # TODO: predefined random data loader
  job["algorithm_parameters"]["N_tests"] = job["job_parameters"]["N_tests"]
  job["algorithm_parameters"]["N_random_init"] = job["job_parameters"]["N_random_init"]
  algorithm_class = algorithm.loadAlgorithm(job["algorithm"])
  algorithm = algorithm_class(sut=sut,
                              test_repository=test_repository,
                              objective_func=objective_func,
                              objective_selector=objective_selector,
                              parameters=job["algorithm_parameters"],
                              logger=logger)

  # Setup the experiment.
  experiment = OwnExperiment(algorithm)

  # Start the experiment.
  experiment.start()

