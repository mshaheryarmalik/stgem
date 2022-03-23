#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Here you can find SUTs relevant to the SBST CPS competition where the
BeamNG.tech car simulator is being tested for faults.
"""

import os, time, traceback
from math import pi, sin, cos
import logging

import numpy as np

# Disable BeamNG logs etc.
for id in ["shapely.geos", "beamngpy.BeamNGpy", "beamngpy.beamng", "beamngpy.Scenario", "beamngpy.Vehicle", "beamngpy.Camera", "matplotlib", "matplotlib.pyplot", "matplotlib.font_manager", "PIL.PngImagePlugin"]:
    logger = logging.getLogger(id)
    logger.setLevel(logging.CRITICAL)
    logger.disabled = True

from shapely.geometry import Point
from shapely.geometry import LineString, Polygon
from shapely.affinity import translate, rotate
from descartes import PolygonPatch

from stgem.sut import SUT
from util import frechet_distance, sbst_validate_test

from self_driving.beamng_brewer import BeamNGBrewer
from self_driving.beamng_car_cameras import BeamNGCarCameras
from self_driving.beamng_tig_maps import maps, LevelsFolder
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.nvidia_prediction import NvidiaPrediction
from self_driving.simulation_data_collector import SimulationDataCollector
from self_driving.utils import get_node_coords, points_distance
from self_driving.vehicle_state_reader import VehicleStateReader

from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.validation import TestValidator
from code_pipeline.visualization import RoadTestVisualizer

class SBSTSUT_base(SUT):
    """
    Implements a base class for SBST based systems under test. The purpose of
    this class is essentially to provide execution on BeamNG.tech with arbitrary
    road input (given as plane points). The actual input representations are
    implemented by subclassing this class and by implementing a function
    transforming the input to a sequence of road points. The execute_test method
    of this class will raise a NotImplementedError.
    """

    def __init__(self, parameters):
        """
        Initialize the class.

        Due to some strange choices in the competition code observe the following
        about paths:
          - You should set beamng_home to point to the directory where the
            simulator was unpacked.
          - The level files (directory levels) are hardcoded to be at
            os.path.join(os.environ["USERPROFILE"], "Documents/BeamNG.research/levels")
          - While the beamng_user parameter of BeamNGBrewer can be anything, it
            makes sense to set it to be the parent directory of the above as it is
            used anyway.
          - The levels_template folder (from the competition GitHub) should be in
            the directory where the code is run from, i.e., it is set to be
            os.path.join(os.getcwd(), "levels_template")

        Args:
          beamng_home (str):      Path to the simulators home directory (i.e.,
                                  where the simulator zip was unpacked; has Bin64
                                  etc. as subdirectories).
          map_size (int):         Map size in pixels (total map map_size*map_size).
          max_speed (float):      Maximum speed (km/h) for the vehicle during the simulation.
          check_key (bool):       Check if the activation key exists.
        """

        super().__init__(parameters)

        if self.map_size <= 0:
            raise ValueError("The map size must be positive.")
        if self.max_speed <= 0:
            raise ValueError("The maximum speed should be positive.")

        # This variable is essentially where (some) files created during the
        # simulation are placed and it is freely selectable. Due to some choices in
        # the SBST CPS competition code, we hard code it as follows (see
        # explanation in the docstring).
        self.beamng_user = os.path.join(
            os.environ["USERPROFILE"], "Documents/BeamNG.research"
        )
        self.oob_tolerance = 0.95  # This is used by the SBST code, but the value does not matter.
        self.max_speed_in_ms = self.max_speed * 0.277778

        # Check for activation key.
        if not os.path.exists(os.path.join(self.beamng_user, "tech.key")):
            raise Exception("The activation key 'tech.key' must be in the directory {}.".format(self.beamng_user))

       # For validating the executed roads.
        self.validator = TestValidator(map_size=self.map_size)

        # Disable log messages from third party code.
        logging.StreamHandler(stream=None)

        # The code below is from the SBST CPS competition.
        # Available at https://github.com/se2p/tool-competition-av
        # TODO This is specific to the TestSubject, we should encapsulate this better
        self.risk_value = 0.7
        # Runtime Monitor about relative movement of the car
        self.last_observation = None
        # Not sure how to set this... How far can a car move in 250 ms at 5Km/h
        self.min_delta_position = 1.0

        # These are set in test execution.
        self.brewer = None
        self.vehicle = None

    def _is_the_car_moving(self, last_state):
        """
        Check if the car moved in the past 10 seconds
        """

        # Has the position changed
        if self.last_observation is None:
            self.last_observation = last_state
            return True

        # If the car moved since the last observation, we store the last state and move one
        if (Point(self.last_observation.pos[0], self.last_observation.pos[1]).distance(Point(last_state.pos[0], last_state.pos[1])) > self.min_delta_position):
            self.last_observation = last_state
            return True
        else:
            # How much time has passed since the last observation?
            if last_state.timer - self.last_observation.timer > 10.0:
                return False
            else:
                return True

    def end_iteration(self):
        try:
            if self.brewer:
                self.brewer.beamng.stop_scenario()
        except Exception as ex:
            traceback.print_exception(type(ex), ex, ex.__traceback__)

    def _execute_test_beamng(self, test):
        """
        Execute a single test on BeamNG.tech and return its output signal. The
        output signal is simply the BLOP (body out of lane percentage) at certain
        time steps. Notice that we expect the input to be a sequence of plane
        points.
        """

        # This code is mainly from https://github.com/se2p/tool-competition-av/code_pipeline/beamng_executor.py

        if self.brewer is None:
            self.brewer = BeamNGBrewer(beamng_home=self.beamng_home, beamng_user=self.beamng_user)
            self.vehicle = self.brewer.setup_vehicle()

        the_test = RoadTestFactory.create_road_test(test)

        # Check if the test is really valid.
        valid, msg = self.validator.validate_test(the_test)
        if not valid:
            # print("Invalid test, not run on SUT.")
            return 0.0

        # For the execution we need the interpolated points
        nodes = the_test.interpolated_points

        brewer = self.brewer
        brewer.setup_road_nodes(nodes)
        beamng = brewer.beamng
        waypoint_goal = BeamNGWaypoint("waypoint_goal", get_node_coords(nodes[-1]))

        # Notice that maps and LevelsFolder are global variables from
        # self_driving.beamng_tig_maps.
        beamng_levels = LevelsFolder(os.path.join(self.beamng_user, "0.24", "levels"))
        maps.beamng_levels = beamng_levels
        maps.beamng_map = maps.beamng_levels.get_map("tig")
        # maps.print_paths()

        maps.install_map_if_needed()
        maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + "\n" + waypoint_goal.to_json())

        vehicle_state_reader = VehicleStateReader(self.vehicle, beamng, additional_sensors=None)
        brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

        steps = brewer.params.beamng_steps
        simulation_id = time.strftime("%Y-%m-%d--%H-%M-%S", time.localtime())
        name = "beamng_executor/sim_$(id)".replace("$(id)", simulation_id)
        sim_data_collector = SimulationDataCollector(
            self.vehicle,
            beamng,
            brewer.decal_road,
            brewer.params,
            vehicle_state_reader=vehicle_state_reader,
            simulation_name=name,
        )

        # TODO: Hacky - Not sure what's the best way to set this...
        sim_data_collector.oob_monitor.tolerance = self.oob_tolerance

        sim_data_collector.get_simulation_data().start()
        try:
            # start = timeit.default_timer()
            brewer.bring_up()
            # iterations_count = int(self.test_time_budget/250)
            # idx = 0

            brewer.vehicle.ai_set_aggression(self.risk_value)
            # Sets the target speed for the AI in m/s, limit means this is the maximum value (not the reference one)
            brewer.vehicle.ai_set_speed(self.max_speed_in_ms, mode="limit")
            brewer.vehicle.ai_drive_in_lane(True)
            brewer.vehicle.ai_set_waypoint(waypoint_goal.name)

            while True:
                # idx += 1
                # assert idx < iterations_count, "Timeout Simulation " + str(sim_data_collector.name)

                sim_data_collector.collect_current_data(oob_bb=True)
                last_state = sim_data_collector.states[-1]
                # Target point reached
                if (points_distance(last_state.pos, waypoint_goal.position) < 8.0):
                    break

                assert self._is_the_car_moving(last_state), "Car is not moving fast enough " + str(sim_data_collector.name)

                assert (not last_state.is_oob), "Car drove out of the lane " + str(sim_data_collector.name)

                beamng.step(steps)

            sim_data_collector.get_simulation_data().end(success=True)
        except AssertionError as aex:
            sim_data_collector.save()
            # An assertion that trigger is still a successful test execution, otherwise it will count as ERROR
            sim_data_collector.get_simulation_data().end(
                success=True, exception=aex
            )
            # traceback.print_exception(type(aex), aex, aex.__traceback__)
        except Exception as ex:
            sim_data_collector.save()
            sim_data_collector.get_simulation_data().end(
                success=False, exception=ex
            )
            traceback.print_exception(type(ex), ex, ex.__traceback__)
        finally:
            sim_data_collector.save()
            try:
                sim_data_collector.take_car_picture_if_needed()
            except:
                pass

            self.end_iteration()

        # Build a time series for the OOB percentage based on simulation states.
        # The time plus OOB percentage is the output signal.
        states = sim_data_collector.get_simulation_data().states
        timestamps = np.zeros(len(states))
        oob = np.zeros(shape=(1, len(states)))
        for i, state in enumerate(states):
            timestamps[i] = state.timer
            oob[0, i] = state.oob_percentage
        return timestamps, oob

class SBSTSUT_curvature(SBSTSUT_base):
    """
    A class to be inherited by all SBST SUTs which use input representation based
    on a fixed number of curvature points. That is, the following input
    representation: input is (c1, ..., ck) where c1, ..., ck are curvature
    values. The step length is fixed (see the method test_to_road_points). All
    inputs are transformed to roads which begin at the middle of the bottom part
    of the map and point initially directly upwards.
    """

    def __init__(self, parameters):
        """
        Args:
          curvature_points (int): How many curvature values specify a road.
          map_size (int): Map size in pixels (total map size map_size*map_size).
        """

        super().__init__(parameters)

        if self.curvature_points <= 0:
            raise ValueError("The number of curvature points must be positive.")
        if self.map_size <= 0:
            raise ValueError("The map size must be positive.")

    def test_to_road_points(self, test):
        """
        Converts a test to road points.

        Args:
          test (list): List of floats in [-1, 1].

        Returns:
          output (list): List of length len(test) of coordinate tuples.
        """

        # This is the same code as in the Frenetic algorithm.
        # https://github.com/ERATOMMSD/frenetic-sbst21/blob/main/src/generators/base_frenet_generator.py
        # We integrate curvature (acceleratation) to get an angle (speed) and then
        # we move one step to this direction to get position. The integration is
        # done using the trapezoid rule with step given by the first component of
        # the test. Previously the first coordinate was normalized back to the
        # interval [25, 35], now we simply fix the step size.
        step = 15
        # We undo the normalization of the curvatures from [-1, 1] to [-0.07, 0.07]
        # as in the Frenetic algorithm.
        curvature = 0.07 * test

        # The initial point is the bottom center of the map. The initial angle is
        # 90 degrees.
        points = [
            (self.map_size / 2, 10)
        ]  # 10 is margin for not being out of bounds
        angles = [np.math.pi / 2]
        # Add the second point.
        points.append( (points[-1][0] + step * np.cos(angles[-1]), points[-1][1] + step * np.sin(angles[-1]) ))
        # Find the remaining points.
        for i in range(curvature.shape[0] - 1):
            angles.append(angles[-1] + step * (curvature[i + 1] + curvature[i]) / 2)
            x = points[-1][0] + step * np.cos(angles[-1])
            y = points[-1][1] + step * np.sin(angles[-1])
            points.append((x, y))

        return points

    def execute_random_test(self):
        """
        Execute a random tests and return it and its output.

        Returns:
          test (np.ndarray): Array of shape (self.curvature_points) of floats in
                             [-1, 1].
          timestamps (np.ndarray): Array of shape (N, 1).
          oob (np.ndarray): Array of shape (1, N).
        """

        test = self.sample_input_space()
        timestamps, oob = self.execute_test(test)
        return test, timestamps, oob

    def _sample_input_space(self, curvature_points):
        """
        Return a sample (test) from the input space.

        Args:
          curvature_points (int): Number of curvature points.

        Returns:
          test (np.ndarray): Array of shape (curvature_points) of floats in
                             [-1, 1].
        """

        if curvature_points <= 0:
            raise ValueError("The number of curvature points must be positive.")

        # The components of the actual test are curvature values in the range
        # [-0.07, 0.07], but the generator output is expected to be in the interval
        # [-1, 1].
        # return np.random.uniform(-1, 1, size=(N, curvature_points))
        #
        # We do not choose the components of a test independently in [-1, 1] but
        # we do as in the case of the Frenetic algorithm where the next component
        # is in the range of the previous value +- 0.05.
        test = np.zeros(curvature_points)
        test[0] = np.random.uniform(-1, 1)
        for i in range(1, curvature_points):
            test[i] = test[i - 1] + (1 / 0.07) * np.random.uniform(-0.05, 0.05)
        return test

    def sample_input_space(self):
        """
        Return a sample (test) from the input space.

        Returns:
          test (np.ndarray): Array of shape (self.curvature_points) of floats in
                             [-1, 1].
        """

        return self._sample_input_space(self.curvature_points)

class SBSTSUT(SBSTSUT_curvature):
    """
    Class for the SBST CPS competition SUT accepting tests which are vectors of
    fixed length.
    """

    def __init__(self, parameters):
        super().__init__(parameters)

    def _execute_test(self, test):
        """
        Execute the given test on the SUT.

        Args:
          test (np.ndarray): Array with shape (1, N) or (N) with N curvature
                             values.

        Returns:
          timestamps (np.ndarray): Array of shape (M, 1).
          oob (np.ndarray): Array of shape (1, M).
        """

        return self._execute_test_beamng(self.test_to_road_points(test.reshape(-1)))

    def validity(self, test):
        """
        Validate the given test.

        Args:
          test (np.ndarray): Array with shape (M) with M curvature values.

        Returns:
          result (float)
        """

        return sbst_validate_test(self.test_to_road_points(test), self.map_size)

    def distance_frechet(self, X, Y):
        """
        Returns the discrete Fréchet distance between the road points defined by
        the tests X and Y.

        Args:
          X (np.ndarray): Test array of shape (1, N) or (N) of floats in [-1, 1].
          Y (np.ndarray): Test array of shape (1, N) or (N) of floats in [-1, 1].

        Returns:
          result (float): The Fréchet distance of X and Y.
        """

        if len(X.shape) > 2 or len(Y.shape) > 2:
            raise ValueError("The tests must be 1- or 2-dimensional arrays.")
        X = X.reshape(-1)
        Y = Y.reshape(-1)
        if X.shape[0] != Y.shape[0]:
            raise ValueError("The tests must have the same length.")

        return frechet_distance(self.test_to_road_points(X), self.test_to_road_points(Y))

class SBSTSUT_validator(SBSTSUT_curvature):
    """
    Class for the SUT of considering an SBST test valid or not. We use the
    following input representation: input is (c1, ..., ck) where c1, ..., ck are
    curvature values. The step length is fixed (see the method
    test_to_road_points). All inputs are transformed to roads which begin at the
    middle of the bottom part of the map and point initially directly upwards.
    """

    def __init__(self, parameters):
        super().__init__(parameters)

    def _execute_test(self, test):
        """
        Execute the given test on the SUT.

        Args:
          test (np.ndarray): Array with shape (1,N) or (N) of N curvature values.

        Returns:
          result (np.ndarray): Array of shape (1).
        """

        return np.array(sbst_validate_test(self.test_to_road_points(test), self.map_size)).reshape(1)

