#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This code collects data from the algorithms WOGAN, Frenetic, and Random of the
SBST 2022 CPS testing competition. The code is called as follows:

  sbst_collect_and_analyze.py frenetic path random path wogan path

Where path is a directory which contains subdirectories with files
session_parameters and training_data.npy in them. Multiple paths can be
specified by repeating 'algorithm path'.
"""

import itertools, json, os, sys
from math import atan2

import numpy as np
import matplotlib.pyplot as plt

from code_pipeline.tests_generation import RoadTestFactory

from config import config, get_model
from sut.sut_sbst import move_road

def collect_data(algorithms, paths, time_budget=None):
  """
  Collect data from the specified paths for specified algorithms. If time_budget is not None, then we trim each
  experiment to take at most time_budget seconds.
  """

  experiments = {a:{"data":[], "test_suites": [], "test_suite_fitnesses": [], "failed": []} for a in algorithms}
  for algorithm in algorithms:
    for path in paths[algorithm]:
      for directory in os.listdir(path):
        directory = os.path.join(path, directory)
        if not os.path.isdir(directory): continue
        if not os.path.exists(os.path.join(directory, "session_parameters")): continue
        with open(os.path.join(directory, "session_parameters")) as f:
          experiments[algorithm]["data"].append(json.load(f))

        data = experiments[algorithm]["data"][-1]

        if algorithm == "frenetic":
          # We remove data related to invalid tests. Moreover, execution times
          # related to invalid tests are added to the generation time of the
          # next valid test.
          new_test_suite = []
          new_max_oob = []
          new_time_generation = []
          new_time_execution = []
          new_time_training = []
          accumulated = 0
          for i in range(len(data["max_oob"])):
            if data["max_oob"][i] >= 0.0:
              new_test_suite.append(data["tests"][i])
              new_max_oob.append(data["max_oob"][i])
              new_time_execution.append(data["time_execution"][i])
              new_time_generation.append(data["time_generation"][i] + accumulated)
              new_time_training.append(data["time_generation"][i])
              accumulated = 0
            else:
              accumulated += data["time_generation"][i] + data["time_execution"][i]

          data["time_execution"] = new_time_execution
          data["time_generation"] = new_time_generation
          data["time_training"] = new_time_training
          data["time_execution_total"] = sum(data["time_execution"])
          data["time_generation_total"] = sum(data["time_generation"])
          data["time_training_total"] = sum(data["time_training"])
          experiments[algorithm]["test_suites"].append(new_test_suite)
          experiments[algorithm]["test_suite_fitnesses"].append(new_max_oob)
          del data["max_oob"]
          del data["tests"]
        else:
          with open(os.path.join(directory, "training_data.npy"), mode="rb") as f:
            X = np.load(f)
            Y = np.load(f)
          tests = []
          fitnesses = []
          for i in range(X.shape[0]):
            tests.append(sut.test_to_road_points(X[i]))
            fitnesses.append(Y[i,0])
          experiments[algorithm]["test_suites"].append(tests)
          experiments[algorithm]["test_suite_fitnesses"].append(fitnesses)
          del X
          del Y

        if not "time_generation_total" in data:
          data["time_generation_total"] = 0.0

        # Trim/expand the experiment if necessary.
        if time_budget is not None:
          data = experiments[algorithm]["data"][-1]

          if data["time_total"] > time_budget:
            # Trim.
            # Find out from which index to trim.
            cum_total = 0.0
            for i in range(len(data["time_execution"])):
              s = cum_total + data["time_execution"][i]
              if i < len(data["time_generation"]):
                s += data["time_generation"][i]
              if i < len(data["time_training"]):
                s += data["time_training"][i]
              if s > time_budget:
                break
              cum_total = s
            if s <= time_budget: i += 1

            # Trim tests and their fitnessess.
            experiments[algorithm]["test_suites"][-1] = experiments[algorithm]["test_suites"][-1][:i]
            experiments[algorithm]["test_suite_fitnesses"][-1] = experiments[algorithm]["test_suite_fitnesses"][-1][:i]
            # Trim times.
            data["time_total"] = cum_total
            for k in ["time_execution", "time_generation", "time_training"]:
              k2 = "{}_total".format(k)
              data[k2] = data[k2] - sum(data[k][i:])
              data[k] = data[k][:i]
          elif algorithm == "random":
            # Expand.
            # Sample new tests randomly from the existing tests until time
            # budget is exhausted.
            L = len(experiments[algorithm]["test_suites"][-1])
            while data["time_total"] <= time_budget:
              idx = np.random.choice(list(range(L)))
              experiments[algorithm]["test_suites"][-1].append(experiments[algorithm]["test_suites"][-1][idx])
              experiments[algorithm]["test_suite_fitnesses"][-1].append(experiments[algorithm]["test_suite_fitnesses"][-1][idx])
              data["time_execution"].append(data["time_execution"][idx])
              data["time_execution_total"] += data["time_execution"][-1]
              data["time_total"] += data["time_execution"][-1]
              data["time_generation"].append(data["time_generation"][idx])
              data["time_generation_total"] += data["time_execution"][-1]
              data["time_total"] += data["time_generation"][-1]
              if idx < len(data["time_training"]):
                data["time_training"].append(data["time_training"][idx])
                data["time_training_total"] += data["time_execution"][-1]
                data["time_total"] += data["time_training"][-1]

        # Average fitnesses.
        # final 80 %
        experiments[algorithm]["test_suite_fitnesses_avg80"] = []
        for i in range(len(experiments[algorithm]["test_suite_fitnesses"])):
          L = len(experiments[algorithm]["test_suite_fitnesses"][i])
          experiments[algorithm]["test_suite_fitnesses_avg80"].append(np.mean(experiments[algorithm]["test_suite_fitnesses"][i][int(0.2*L):]))
        # final 20 %
        experiments[algorithm]["test_suite_fitnesses_avg20"] = []
        for i in range(len(experiments[algorithm]["test_suite_fitnesses"])):
          L = len(experiments[algorithm]["test_suite_fitnesses"][i])
          experiments[algorithm]["test_suite_fitnesses_avg20"].append(np.mean(experiments[algorithm]["test_suite_fitnesses"][i][int(0.8*L):]))

        # Number of failed tests.
        failed = len([f for f in experiments[algorithm]["test_suite_fitnesses"][-1] if f >= sut.target])
        experiments[algorithm]["failed"].append(failed)


  return experiments

available_algorithms = ["random", "wogan", "frenetic"]
algorithm_names = {"random": "Random",
                   "wogan": "WOGAN",
                   "frenetic": "Frenetic"}

def interpolate_points(test):
  return RoadTestFactory.create_road_test(test).interpolated_points

if __name__ == "__main__":
  # Parse the command line arguments.
  # ---------------------------------------------------------------------------
  algorithms = []
  paths = {}
  for i in range(1, len(sys.argv), 2):
    algorithm = sys.argv[i]
    path = sys.argv[i+1]
    if not algorithm in available_algorithms:
      raise SystemExit("Unknown algorithm '{}'.".format(algorithm))
    if not algorithm in algorithms:
      algorithms.append(algorithm)
    if not os.path.exists(path):
      raise SystemExit("Path '{}' doesn't exist.".format(path))
    if not algorithm in paths:
      paths[algorithm] = [path]
    else:
      paths[algorithm].append(path)

  # Get the SUT.
  # ---------------------------------------------------------------------------
  model, _, _ = get_model("sbst", "wgan", None)
  sut = model.sut

  # Collect data.
  # ---------------------------------------------------------------------------
  # Time budget (seconds) can be used to trim down a execution based experiment
  # to a time based one. The random algorithm results are expanded by
  # resampling if necessary.
  time_budget = None
  #time_budget = 2*3600
  experiments = collect_data(algorithms, paths, time_budget=time_budget)

  # Draw figures etc.
  # ---------------------------------------------------------------------------
  #
  # Failed tests
  # ---------------------------------------------------------------------------
  data = [experiments[algorithm]["failed"] for algorithm in algorithms]
  labels = [algorithm_names[algorithm] for algorithm in algorithms]
  bp = plt.boxplot(data, labels=labels, patch_artist=True)
  plt.setp(bp["medians"], color="black")
  plt.setp(bp["boxes"], facecolor="white")
  #plt.title("Failed tests")
  plt.xlabel("")
  plt.ylabel("Number of failed tests")
  plt.show()
  plt.clf()

  print("Failed tests:")
  print("-------------")
  for algorithm in algorithms:
    print("{}:".format(algorithm))
    data = [len(e) for e in experiments[algorithm]["test_suites"]]
    print("avg executed tests: {}".format(np.mean(data)))
    print("sd executed tests: {}".format(np.std(data)))
    data = experiments[algorithm]["failed"]
    print("avg of failed tests: {}".format(np.mean(data)))
    print("sd of failed tests: {}".format(np.std(data)))
    data = experiments[algorithm]["test_suite_fitnesses_avg80"]
    print("avg fitness of final 80 %: {}".format(np.mean(data)))
    print("sd fitness of final 80 %: {}".format(np.std(data)))
    data = experiments[algorithm]["test_suite_fitnesses_avg20"]
    print("avg fitness of final 20 %: {}".format(np.mean(data)))
    print("sd fitness of final 20 %: {}".format(np.std(data)))
    print()

  # Time
  # ---------------------------------------------------------------------------
  print("Times:")
  print("------")
  for algorithm in algorithms:
    print("{}:".format(algorithm))

    data = [e["time_total"] for e in experiments[algorithm]["data"]]
    avg_time_total = np.mean(data) / 3600
    std_time_total = np.std(data) / 3600
    print("totals:")
    print("    {} h (std = {} h)".format(round(avg_time_total, 2), round(std_time_total, 2)))

    data = [e["time_execution_total"] for e in experiments[algorithm]["data"]]
    avg_time_execution_total = np.mean(data) / 3600
    std_time_execution_total = np.std(data) / 3600
    print("  execution:")
    print("    {} h (std = {} h)".format(round(avg_time_execution_total, 2), round(std_time_execution_total, 2)))

    if algorithm == "frenetic":
      data = [e["time_generation_total"] for e in experiments[algorithm]["data"]]
    else:
      data = []
      for e in experiments[algorithm]["data"]:
        # We haven't always saved the total, so we sum even if it's unnecessary.
        t = sum(e["time_generation"])
        data.append(e["time_training_total"] + t)
    avg_time_te_total = np.mean(data) / 3600
    std_time_te_total = np.std(data) / 3600
    print("  training + generation:")
    print("    {} h (std = {} h)".format(round(avg_time_te_total, 2), round(std_time_te_total, 2)))

    data = []
    for e in experiments[algorithm]["data"]:
      data.append(sum(e["time_training"]))
    avg_time_te_total = np.mean(data) / 3600
    std_time_te_total = np.std(data) / 3600
    print("  training:")
    print("    {} h (std = {} h)".format(round(avg_time_te_total, 2), round(std_time_te_total, 2)))
    print()

    print("per test:")
    for e in experiments[algorithm]["data"]:
      data += e["time_execution"]
    avg_time_execution = np.mean(data)
    std_time_execution = np.std(data)
    print("  execution:")
    print("    {} s (std = {} s)".format(round(avg_time_execution, 2), round(std_time_execution, 2)))

    if algorithm == "frenetic":
      data = []
      for e in experiments[algorithm]["data"]:
        data += e["time_generation"]
    else:
      data = []
      for e in experiments[algorithm]["data"]:
        data += e["time_generation"]
        data += e["time_training"]
    avg_time_te = np.mean(data)
    std_time_te = np.std(data)
    print("  training + generation:")
    print("    {} s (std = {} s)".format(round(avg_time_te, 2), round(std_time_te, 2)))

    data = []
    for e in experiments[algorithm]["data"]:
      data += e["time_training"]
    if len(data) > 0:
      avg_time_te = np.mean(data)
      std_time_te = np.std(data)
    else:
      avg_time_te = 0
      std_time_te = 0
    print("  training:")
    print("    {} s (std = {} s)".format(round(avg_time_te, 5), round(std_time_te, 5)))
    print()

  # Normalize the failed tests in test suites
  # ---------------------------------------------------------------------------
  # We have determined in advance that when six plane points are used, they are
  # interpolated to 75 points for the simulator. We thus adjust all
  # interpolations to this many points. Then we normalize the points to all
  # start from the origin and point initially upwards.
  points = 75
  for algorithm in algorithms:
    experiments[algorithm]["test_suites_failed_normalized"] = []
    for n, test_suite in enumerate(experiments[algorithm]["test_suites"]):
      normalized_test_suite = []
      # Interpolate all tests and adjust the length of the list of interpolated
      # points.
      for m, test in enumerate(test_suite):
        if experiments[algorithm]["test_suite_fitnesses"][n][m] < sut.target: continue
        # Interpolate.
        interpolated = interpolate_points(test)
        # Adjust length.
        idx = np.round(np.linspace(0, len(interpolated) - 1, points)).astype(int)
        adjusted = list(np.array(interpolated)[idx])
        # Normalize.
        normalized_test_suite.append(move_road(adjusted, 0, 0))

      experiments[algorithm]["test_suites_failed_normalized"].append(normalized_test_suite)

  # Diversity of failed tests
  # ---------------------------------------------------------------------------
  for algorithm in algorithms:
    experiments[algorithm]["test_suite_failed_diversities_euclidean"] = []
    for test_suite in experiments[algorithm]["test_suites_failed_normalized"]:
      # Convert the test suite to angles.
      all_tests_angles = []
      for test in test_suite:
        test_angle = []
        for i in range(1, len(test)):
          dx = test[i][0] - test[i-1][0]
          dy = test[i][1] - test[i-1][1]
          test_angle.append(atan2(dy, dx))
        all_tests_angles.append(test_angle)

      # Compute pairwise Euclidean distances for the test suite
      euclidean_distances = [np.linalg.norm(np.array(t1) - np.array(t2)) for t1, t2 in itertools.combinations(all_tests_angles, 2)]

      # Save the median Euclidean distance for the test suite.
      experiments[algorithm]["test_suite_failed_diversities_euclidean"].append(np.median(euclidean_distances))

  data = [experiments[algorithm]["test_suite_failed_diversities_euclidean"] for algorithm in algorithms]
  labels = [algorithm_names[algorithm] for algorithm in algorithms]
  bp = plt.boxplot(data, labels=labels, patch_artist=True)
  plt.setp(bp["medians"], color="black")
  plt.setp(bp["boxes"], facecolor="white")
  #plt.title("Test diversity Euclidean")
  plt.xlabel("")
  plt.ylabel("Diversity")
  plt.show()
  plt.clf()

  for algorithm in algorithms:
    data = experiments[algorithm]["test_suite_failed_diversities_euclidean"]
    print("{}:".format(algorithm))
    print("avg diversity of failed tests: {}".format(np.mean(data)))
    print("sd diversity of failed tests: {}".format(np.std(data)))
    print()
