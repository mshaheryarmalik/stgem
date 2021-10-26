#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, datetime

import imageio

import numpy as np
import torch

from config import *
from validator.validator import Validator
from models import OGAN, RandomGenerator

def log(msg):
  print(msg)

if __name__ == "__main__":
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
  # odroid_ogan, sbst_validator_ogan, sbst_ogan
  sut_id = "odroid_ogan"

  enable_log = True
  enable_view = True
  enable_save = True

  # Initialize the system under test and validator.
  # ---------------------------------------------------------------------------
  # We assume that there exists an efficient and perfect validator oracle. This
  # oracle is used mainly for test validation, and the trained proxy for it is
  # only used for training.
  if sut_id == "odroid_ogan":
    from sut.sut_odroid import OdroidSUT

    random_init = 50
    N = 200

    output_data = 1
    fitness_threshold = 6.0

    sut = OdroidSUT(output_data, fitness_threshold)
    validator = None

    epoch_settings_init = {"epochs": 2,
                           "analyzer_epochs": 20,
                           "critic_epochs": 5,
                           "generator_epochs": 1}
    epoch_settings = {"epochs": 1,
                      "analyzer_epochs": 5,
                      "critic_epochs": 5,
                      "generator_epochs": 1}

    def _view_test(test):
      pass

    def _save_test(test, session, file_name):
      pass

  elif sut_id == "sbst_validator_ogan":
    from sut.sut_sbst import SBSTSUT_beamng, SBSTSUT_validator, sbst_test_to_image, sbst_validate_test

    random_init = 50
    N = 200

    map_size = 200
    curvature_points = 5

    validator_bb = Validator(input_size=curvature_points, validator_bb=lambda t: sbst_validate_test(t, sut))
    sut = SBSTSUT_validator(map_size=map_size, curvature_points=validator_bb.ndimensions, validator_bb=validator_bb)
    test_to_image = lambda t, file_name=None: _test_to_image(sut, t.reshape(sut.ndimensions), file_name)
    validator = None

    epoch_settings_init = {"epochs": 2,
                           "analyzer_epochs": 20,
                           "critic_epochs": 5,
                           "generator_epochs": 1}
    epoch_settings = {"epochs": 1,
                      "analyzer_epochs": 5,
                      "critic_epochs": 5,
                      "generator_epochs": 1}

    def _view_test(test):
      plt = sbst_test_to_image(convert(test), sut)
      plt.show()

    def _save_test(test, session, file_name):
      plt = sbst_test_to_image(convert(test), sut)
      plt.savefig(os.path.join(config["test_save_path"], sut_id, session, file_name + ".jpg"))

  elif sut_id == "sbst_ogan":
    from sut.sut_sbst import SBSTSUT_beamng, SBSTSUT_validator, sbst_test_to_image, sbst_validate_test

    random_init = 50
    N = 200

    map_size = 200
    curvature_points = 5

    sut = SBSTSUT_beamng(config["sbst"]["beamng_home"], map_size=map_size, curvature_points=curvature_points)
    validator = Validator(sut.ndimensions, lambda t: sbst_validate_test(t, sut))

    epoch_settings_init = {"epochs": 2,
                           "analyzer_epochs": 20,
                           "critic_epochs": 5,
                           "generator_epochs": 1}
    epoch_settings = {"epochs": 1,
                      "analyzer_epochs": 5,
                      "critic_epochs": 5,
                      "generator_epochs": 1}

    def _view_test(test):
      plt = sbst_test_to_image(convert(test), sut)
      plt.show()

    def _save_test(test, session, file_name):
      plt = sbst_test_to_image(convert(test), sut)
      plt.savefig(os.path.join(config["test_save_path"], sut_id, session, file_name + ".jpg"))

  else:
    print("No SUT specified.")
    raise SystemExit

  session = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
  view_test = lambda t: _view_test(t) if enable_view else None
  save_test = lambda t, f: _save_test(t, session, f) if enable_save else None

  os.makedirs(os.path.join(config["test_save_path"], sut_id, session), exist_ok=True)

  # Initialize the model.
  # ---------------------------------------------------------------------------
  model = OGAN(sut, validator, device)
  #model = RandomGenerator(sut, validator, device)

  """
  Here we begin to sample new tests from the input space and begin the
  generation of new tests. The SUT methods return numpy arrays, but we convert
  the tests to lists of tuples (which are hashable) and maintain a dictionary
  of used tests for efficiency.
  """

  # Generate initial tests randomly.
  # ---------------------------------------------------------------------------
  # TODO: We should check that no test is obtained twice.
  test_inputs = []
  test_outputs = []
  test_visited = {}
  roads_similarity_tolerance = 0.05

  def detect_similar_roads(test_n):

    for test_v in test_visited:
      # distance compare option 1
      difference = np.sqrt(np.sum((np.array(test_v) - np.array(test_n)) ** 2))

      # distance compare option 2
      # difference = np.sqrt(np.sum(np.abs((np.array(test_v) - np.array(test_n)))))

      # distance compare option 3
      # difference = np.max(np.abs(np.array(test_v) - np.array(test_n)))

      if difference <= roads_similarity_tolerance:
        # print('too similar -----------------------------------------------------------', difference)
        return True

      else:
        continue

  load = False
  if load:
    log("Loading pregenerated initial tests.")
    data = np.load(config[sut_id]["pregenerated_initial_data"]).tolist()
    for test in data:
      test_inputs.append(tuple(test[:-1]))
      test_outputs.append(test[-1])
      test_visited[tuple(test[:-1])] = True
    del data
  else:
    log("Generating and running {} random valid tests.".format(random_init))
    while len(test_inputs) < random_init:
      test = sut.sample_input_space()
      if model.validity(test)[0,0] == 0: continue
      test_inputs.append(tuple(test[0,:]))

      view_test(test)
      save_test(test, "init_{}".format(len(test_inputs)))

      log("Executing {} ({}/{})".format(test, len(test_inputs), random_init))
      test_outputs.append(sut.execute_test(test)[0,0])
      log("Result: {}".format(test_outputs[-1]))
      test_visited[tuple(test[0,:])] = True

  # Train the model with initial tests.
  # ---------------------------------------------------------------------------
  log("Training model...")
  model.train_with_batch(np.array(test_inputs),
                         np.array(test_outputs).reshape(len(test_outputs), 1),
                         epoch_settings=epoch_settings)

  # Begin the main loop for new test generation and training.
  # ---------------------------------------------------------------------------
  # How many tests are generated.
  while len(test_inputs) < N:
    # Generate a new valid test with high fitness and decrease target fitness
    # as per execution of the loop.
    log("Starting to generate a new test.")
    target_fitness = 1
    rounds = 0
    invalid = 0
    while True:
      # Generate a new valid test (from noise), but do not use a test that has
      # already been used.
      new_test = model.generate_test()
      # TODO: in order to traverse the test space more completely, we probably
      #       should exclude tests that are "too close" to tests already
      #       generated. Ivan's code does this.
      #if tuple(new_test[0,:]) in test_visited: continue

      if detect_similar_roads(tuple(new_test[0,:])):
        continue

      # Check if the test is valid.
      if model.validity(new_test)[0,0] == 0:
        invalid += 1
        continue

      # Predict the fitness of the new test.
      new_fitness = model.predict_fitness(new_test)[0,0]

      target_fitness *= 0.95 # TODO: make configurable

      # Check if the new test has high enough fitness.
      if new_fitness >= target_fitness: break
      rounds += 1

    log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(new_test, new_fitness, rounds + 1, invalid))
    view_test(new_test)
    save_test(new_test, "test_{}".format(len(test_inputs) + 1))

    # Add the new test to our test suite.
    test_inputs.append(tuple(new_test[0,:]))
    test_visited[tuple(new_test[0,:])] = True
    # Actually run the new test on the SUT.
    log("Executing the test...")
    test_outputs.append(model.sut.execute_test(new_test)[0,0])
    log("The actual fitness {} for the generated test.".format(test_outputs[-1]))

    # Train the model with the new test.
    log("Training the model...")
    model.train_with_batch(np.array(test_inputs),
                           np.array(test_outputs).reshape(len(test_outputs), 1),
                           epoch_settings=epoch_settings_init)

  # Evaluate the generated tests.
  # ---------------------------------------------------------------------------
  total = len(test_inputs)
  log("Generated total {} tests.".format(total))

  total_positive = sum(1 for output in test_outputs if output >= sut.target)
  log("{}/{} ({} %) are positive.".format(total_positive, total, round(total_positive/total*100, 1)))

  fitness = model.predict_fitness(np.array(test_inputs))
  total_predicted_positive = sum(fitness >= target_fitness)[0]
  log("{}/{} ({} %) are predicted to be positive".format(total_predicted_positive, total, round(total_predicted_positive/total*100, 1)))

  # Generate new samples to assess quality visually.
  for n, test in enumerate(model.generate_test(30)):
    view_test(test)
    save_test(test, "eval_{}".format(n + 1))

  # Create an animation out of the generated roads.
  if sut_id in ["sbst_validator_ogan", "sbst_ogan"]:
    session_directory = os.path.join(config["test_save_path"], sut_id, session)
    training_images = []
    eval_images = []
    for filename in os.listdir(session_directory):
      data = imageio.imread(os.path.join(session_directory, filename))
      if filename.startswith("init") or filename.startswith("test"):
        training_images.append(data)
      elif filename.startswith("eval"):
        eval_images.append(data)

    imageio.mimsave(os.path.join(session_directory, "training_clip.gif"), training_images)
    imageio.mimsave(os.path.join(session_directory, "eval_clip.gif"), eval_images)

