#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

from time import sleep

import numpy as np
import torch

from config import *
from sut.sut_odroid import OdroidSUT
from sut.sut_sbst import SBSTSUT_beamng, SBSTSUT_validator, sbst_test_to_image, sbst_validate_test
from validator.validator import Validator
from models import GAN, RandomGenerator
from datetime import datetime
import imageio

cwd = os.getcwd()
parent_dir_path = os.path.abspath(os.path.join(cwd, os.pardir))
session_folder_name = "tmp-" + datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
session_directory = os.path.join(parent_dir_path, "simulations", session_folder_name)

def log(msg):
    print(msg)

def collect_initial_training_data(model, N, file_name, append=True):
    if append and os.path.exists(file_name + ".npy"):
        old = np.load(file_name + ".npy")
        data = np.zeros(shape=(old.shape[0] + N, model.sut.ndimensions + 1))
        data[0:old.shape[0],:] = old[0:old.shape[0],:]
        offset = old.shape[0]
    else:
        data = np.zeros(shape=(N, model.sut.ndimensions + 1))
        offset = 0

    for n in range(N):
        test = sut.sample_input_space()
        if model.validity(test)[0, 0] == 0.0: continue
        data[offset + n,0:-1] = test
        data[offset + n,-1] = model.sut.execute_test(test)[0,0]
        # Without this we can get a connection timeout with the simulator.
        sleep(5)

    np.save(file_name, data)

def _test_to_image(sut, test, file_name=None):
    plt = sbst_test_to_image(test, sut)
    if file_name is None:
        plt.show()
    else:
        plt.savefig(file_name)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize the system under test and validator.
    # ---------------------------------------------------------------------------
    # We assume that there exists an efficient and perfect validator oracle. This
    # oracle is used mainly for test validation, and the trained proxy for it is
    # only used for training.
    #
    # Odroid.
    """
    output_data = 1
    fitness_threshold = 6.0
    sut = OdroidSUT(output_data, fitness_threshold)
    validator = None
    # SBST competition validator.
    """
    validator_bb = Validator(input_size=5, validator_bb=lambda t: sbst_validate_test(t, sut), device=device)
    sut = SBSTSUT_validator(map_size=200, curvature_points=validator_bb.ndimensions, validator_bb=validator_bb)
    test_to_image = lambda t, file_name=None: _test_to_image(sut, t.reshape(sut.ndimensions), file_name)
    validator = None

    # SBST competetition.
    """
    sut = SBSTSUT_beamng(config["sbst"]["beamng_home"], map_size=200, curvature_points=5)
    validator = Validator(sut.ndimensions, lambda t: sbst_validate_test(t, sut), device=device)
    """

    # Initialize the model.
    # ---------------------------------------------------------------------------
    model = GAN(sut, validator, device)
    #model = RandomGenerator(sut, validator, device)

    """
    Here we begin to sample new tests from the input space and begin the
    generation of new tests. The SUT methods return numpy arrays, but we convert
    the tests to lists of tuples (which are hashable) and maintain a dictionary
    of used tests for efficiency.
    """

    # Generate initial tests randomly.
    # ---------------------------------------------------------------------------
    random_init = 50 # TODO: put to config
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
            # difference = np.sqrt(np.sum((np.array(test_v) - np.array(test_n))))

            # distance compare option 3
            # difference = np.linalg.norm(np.array(test_v) - np.array(test_n))

            if difference <= roads_similarity_tolerance:
                # print('too similar -----------------------------------------------------------', difference)
                return True

            else:
                continue

    """
    for n in range(3):
      collect_initial_training_data(model, 1, "foobar", append=True)
    raise SystemExit
    """

    load = False
    if load:
        log("Loading pregenerated initial tests.")
        data = np.load(config["sbst"]["pregenerated_initial_data"]).tolist()
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
            log("Executing {}".format(test))
            test_outputs.append(sut.execute_test(test)[0,0])
            log("Result: {}".format(test_outputs[-1]))
            test_visited[tuple(test[0,:])] = True

    # Train the model with initial tests.
    # ---------------------------------------------------------------------------
    log("Training model...")
    model.train_with_batch(np.array(test_inputs),
                           np.array(test_outputs).reshape(len(test_outputs), 1),
                           epochs=4,
                           generator_epochs=6,
                           validator_epochs=1,
                           discriminator_epochs=20,
                           validator_data_size=10,
                           discriminator_data_size=10)

    # Begin the main loop for new test generation and training.
    # ---------------------------------------------------------------------------
    # How many tests are generated.
    N = 100 # TODO: put to config/parameter
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

            # if tuple(new_test[0,:]) in test_visited:
            #    continue

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

        os.makedirs(session_directory, exist_ok=True)
        log("Chose test {} with predicted fitness {}. Generated total {} tests of which {} were invalid.".format(new_test, new_fitness, rounds + 1, invalid))
        tmp_image_loc = os.path.join(session_directory, "tmp_{}.jpg".format(len(test_inputs)))
        test_to_image(new_test.reshape((sut.ndimensions)), tmp_image_loc)

        # Add the new test to our test suite.
        test_inputs.append(tuple(new_test[0,:]))
        test_visited[tuple(new_test[0,:])] = True
        # Actually run the new test on the SUT.
        test_outputs.append(model.sut.execute_test(new_test)[0,0])
        log("The actual fitness {} for the generated test.".format(test_outputs[-1]))

        # Train the model with the new test.
        # Set use_final = 1 to train with the new test only.
        log("Training the model...")
        model.train_with_batch(np.array(test_inputs),
                               np.array(test_outputs).reshape(len(test_outputs), 1),
                               epochs=4,
                               generator_epochs=2,
                               validator_epochs=1,
                               discriminator_epochs=15,
                               validator_data_size=10,
                               discriminator_data_size=10,
                               use_final=-1)

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
        new_image_loc = os.path.join(session_directory, "new_{}.jpg".format(n+1))
        test_to_image(test.reshape((sut.ndimensions)), new_image_loc)


    dirs = os.listdir(session_directory)
    images = []
    for filename in dirs:
        im_path = os.path.join(session_directory, filename)
        images.append(imageio.imread(im_path))

    road_clip = os.path.join(session_directory, 'roads_clip.gif')
    imageio.mimsave(road_clip, images)

    os.startfile(road_clip)
