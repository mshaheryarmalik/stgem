#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os, time, traceback
from math import atan2, pi, degrees

import numpy as np

from matplotlib import pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import LineString, Polygon
from shapely.affinity import translate, rotate
from descartes import PolygonPatch

from sut.sut import SUT

from self_driving.beamng_brewer import BeamNGBrewer
from self_driving.beamng_tig_maps import maps, LevelsFolder
from self_driving.beamng_waypoint import BeamNGWaypoint
from self_driving.simulation_data_collector import SimulationDataCollector
from self_driving.utils import get_node_coords, points_distance
from self_driving.vehicle_state_reader import VehicleStateReader
from code_pipeline.tests_generation import RoadTestFactory
from code_pipeline.validation import TestValidator
from code_pipeline.visualization import RoadTestVisualizer

from shapely.geometry import Point

class SBSTSUT(SUT):
  """
  Implements a base class for SBST based systems under test.
  """

  def __init__(self, map_size):
    """
    Args:
      map_size (int): Map size in pixels (total map size map_size*map_size).
    """

    if map_size <= 0:
      raise ValueError("The map size must be positive.")

    super().__init__()
    self.map_size = map_size

  def test_to_road_points(self, test):
    """
    Converts a test instance to road points.

    Args:
      test (list): List of length self.ndimensions of floats in [-1, 1].

    Returns:
      output (list): List of length self.ndimensions of coordinate tuples.
    """

    if len(test) != self.ndimensions:
      raise ValueError("Input list expected to have length {}.".format(self.ndimensions))

    # This is the same code as in the Frenetic algorithm.
    # https://github.com/ERATOMMSD/frenetic-sbst21/blob/main/src/generators/base_frenet_generator.py
    # We integrate curvature (acceleratation) to get an angle (speed) and then
    # we move one step to this direction to get position. The integration is
    # done using the trapezoid rule with step given by the first component of
    # the test. We need to undo the normalization of the first coordinate back
    # to the interval [25, 35].
    step = 5*test[0] + 30
    # We also undo the normalization of the curvatures from [-1, 1] to
    # [-0.07, 0.07] as in the Frenetic algorithm.
    curvature = 0.07*test[1:]

    # The initial point is the bottom center of the map. The initial angle is
    # 90 degrees.
    points = [(self.map_size/2, 10)] # 10 is margin for not being out of bounds
    angles = [np.math.pi/2]
    # Add the second point.
    points.append((points[-1][0] + step*np.cos(angles[-1]), points[-1][1] + step*np.sin(angles[-1])))
    # Find the remaining points.
    for i in range(curvature.shape[0] - 1):
      angles.append(angles[-1] + step*(curvature[i+1] - curvature[i])/2)
      x = points[-1][0] + step*np.cos(angles[-1])
      y = points[-1][1] + step*np.sin(angles[-1])
      points.append((x, y))

    return points

  def _sample_input_space(self, N, curvature_points):
    """
    Return n samples (tests) from the input space.

    Args:
      N (int):                The number of tests to be sampled.
      curvature_points (int): Number of points on a road (test).

    Returns:
      tests (np.ndarray): Array of shape (N, curvature_points).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")
    if curvature_points < 2:
      raise ValueError("The roads must have at least two points.")

    # The first component of a test is segment length for the road between two
    # curvature points. It is modeled as a number in the interval [25, 35]
    # arbitrary choice. The remaining components are curvature values which are
    # currently set to be in [-0.07, 0.07]. The generator is expected to output
    # values in the interval [-1, 1], so the generated tests are normalized to
    # this interval.
    return np.random.uniform(-1, 1, size=(N, curvature_points))

class SBSTSUT_beamng(SBSTSUT):
  """
  Implements the system under test for the BeamNG simulator.
  """

  def __init__(self, beamng_home, map_size, curvature_points, oob_tolerance=0.95, max_speed=70):
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
      curvature_points (int): How many road points are generated.
      oob_tolerance (float):  What percentage of car out of the road is considered a failed test.
      max_speed (float):      Maximum speed for the vehicle during the simulation.
    """

    if map_size <= 0:
      raise ValueError("The map size must be positive.")
    if curvature_points < 2:
      raise ValueError("The roads must have at least two points.")
    if not (0.0 <= oob_tolerance <= 1.0):
      raise ValueError("The oob_tolerance must be between 0.0 and 1.0.")
    if max_speed <= 0:
      raise ValueError("The maximum speed should be positive.")

    super().__init__(map_size)

    # TODO: Some constants and the first coordinate of the test instance
    #       should be proportional to map_size in some sense.

    self.beamng_home = beamng_home
    # This variable is essentially where (some) files created during the
    # simulation are placed and it is freely selectable. Due to some choices in
    # the competition code, we hard code it as follows (see explanation in
    # the docstring).
    self.beamng_user = os.path.join(os.environ["USERPROFILE"], "Documents/BeamNG.research")
    self.map_size = map_size
    self.ndimensions = curvature_points
    self.oob_tolerance = oob_tolerance
    self.target = self.oob_tolerance
    self.maxspeed = max_speed

    # Check for activation key.
    if not os.path.exists(os.path.join(self.beamng_user, "research.key")):
      raise Exception("The activation key 'research.key' must be in the directory {}.".format(self.beamng_user))

    # For validating the executed roads.
    self.validator = TestValidator(map_size=self.map_size)

    # The code below is from the SBST competition.

    # TODO This is specific to the TestSubject, we should encapsulate this better
    self.risk_value = 0.7
    # Runtime Monitor about relative movement of the car
    self.last_observation = None
    # Not sure how to set this... How far can a car move in 250 ms at 5Km/h
    self.min_delta_position = 1.0

    # Setup the brewer and vehicle.
    self.brewer = BeamNGBrewer(beamng_home=self.beamng_home, beamng_user=self.beamng_user)
    self.vehicle = self.brewer.setup_vehicle()

    # Setup maps.
    # Notice that map and LevelsFolder are global variables from
    # self_driving.beamng_tig_maps.
    beamng_levels = LevelsFolder(os.path.join(self.beamng_user, 'levels'))
    maps.beamng_levels = beamng_levels
    maps.beamng_map = maps.beamng_levels.get_map('tig')
    # maps.print_paths()
    maps.install_map_if_needed()

  def _is_the_car_moving(self, last_state):
    """
    Check if the car moved in the past 10 seconds
    """

    # Has the position changed
    if self.last_observation is None:
      self.last_observation = last_state
      return True

    # If the car moved since the last observation, we store the last state and move one
    if Point(self.last_observation.pos[0],self.last_observation.pos[1]).distance(Point(last_state.pos[0], last_state.pos[1])) > self.min_delta_position:
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

  def _execute_single_test(self, test):
    # This code is mainly from https://github.com/se2p/tool-competition-av/code_pipeline/beamng_executor.py
    # Some initialization parts are moved to __init__.
    the_test = RoadTestFactory.create_road_test(self.test_to_road_points(test))

    # Check if the test is really valid.
    valid, msg = self.validator.validate_test(the_test)
    if not valid:
      #print("Invalid test, not run on SUT.")
      return 0.0

    # For the execution we need the interpolated points
    nodes = the_test.interpolated_points

    brewer = self.brewer
    brewer.setup_road_nodes(nodes)
    beamng = brewer.beamng
    waypoint_goal = BeamNGWaypoint('waypoint_goal', get_node_coords(nodes[-1]))

    maps.beamng_map.generated().write_items(brewer.decal_road.to_json() + '\n' + waypoint_goal.to_json())

    vehicle_state_reader = VehicleStateReader(self.vehicle, beamng, additional_sensors=None)
    brewer.vehicle_start_pose = brewer.road_points.vehicle_start_pose()

    steps = brewer.params.beamng_steps
    simulation_id = time.strftime('%Y-%m-%d--%H-%M-%S', time.localtime())
    name = 'beamng_executor/sim_$(id)'.replace('$(id)', simulation_id)
    sim_data_collector = SimulationDataCollector(self.vehicle,
                                                 beamng,
                                                 brewer.decal_road,
                                                 brewer.params,
                                                 vehicle_state_reader=vehicle_state_reader,
                                                 simulation_name=name)

    # TODO: Hacky - Not sure what's the best way to set this...
    sim_data_collector.oob_monitor.tolerance = self.oob_tolerance

    sim_data_collector.get_simulation_data().start()
    try:
      #start = timeit.default_timer()
      brewer.bring_up()
      # iterations_count = int(self.test_time_budget/250)
      # idx = 0

      brewer.vehicle.ai_set_aggression(self.risk_value)
      # TODO This does not seem to take any effect...
      brewer.vehicle.ai_set_speed(self.maxspeed, mode='limit')
      brewer.vehicle.ai_drive_in_lane(True)
      brewer.vehicle.ai_set_waypoint(waypoint_goal.name)

      while True:
        # idx += 1
        # assert idx < iterations_count, "Timeout Simulation " + str(sim_data_collector.name)

        sim_data_collector.collect_current_data(oob_bb=True)
        last_state = sim_data_collector.states[-1]
        # Target point reached
        if points_distance(last_state.pos, waypoint_goal.position) < 8.0:
          break

        assert self._is_the_car_moving(last_state), "Car is not moving fast enough " + str(sim_data_collector.name)

        assert not last_state.is_oob, "Car drove out of the lane " + str(sim_data_collector.name)

        beamng.step(steps)

      sim_data_collector.get_simulation_data().end(success=True)
    except AssertionError as aex:
      sim_data_collector.save()
      # An assertion that trigger is still a successful test execution, otherwise it will count as ERROR
      sim_data_collector.get_simulation_data().end(success=True, exception=aex)
      #traceback.print_exception(type(aex), aex, aex.__traceback__)
    except Exception as ex:
      sim_data_collector.save()
      sim_data_collector.get_simulation_data().end(success=False, exception=ex)
      traceback.print_exception(type(ex), ex, ex.__traceback__)
    finally:
      sim_data_collector.save()
      try:
        sim_data_collector.take_car_picture_if_needed()
      except:
        pass

      self.end_iteration()

    # TODO: We could return other sorts of data as well.

    # Return the highest OOB percentage from the states of the vehicle during
    # the simulation (higher -> more out of bounds). The current competition
    # code has a bug, so we cannot directly return the value of
    # sim_data_collector.get_simulation_data().states[-1].max_oob_percentage
    return max(state.oob_percentage for state in sim_data_collector.get_simulation_data().states)

  def execute_test(self, tests):
    """
    Execute the given tests on the SUT.

    Args:
      tests (np.ndarray): Array of N tests with shape (N, self.ndimensions).

    Returns:
      result (np.ndarray): Array of shape (N, 1).
    """

    if len(tests.shape) != 2 or tests.shape[1] != self.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.ndimensions))

    result = np.zeros(shape=(tests.shape[0], 1))
    for n, test in enumerate(tests):
      result[n,0] = self._execute_single_test(tests[n,:])

    return result

  def execute_random_test(self, N=1):
    """
    Execute N random tests and return their outputs.

    Args:
      N (int): The number of tests to be executed.

    Returns:
      tests (np.ndarray):   Array of shape (N, self.ndimensions).
      outputs (np.ndarray): Array of shape (N, 1).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    dataX = self.sample_input_space(N)
    dataY = self.execute_test(dataX)
    return dataX, dataY

  def sample_input_space(self, N=1):
    """
    Return n samples (tests) from the input space.

    Args:
      N (int): The number of tests to be sampled.

    Returns:
      tests (np.ndarray): Array of shape (N, self.ndimensions).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    return self._sample_input_space(N, self.ndimensions)

class SBSTSUT_validator(SBSTSUT):

  def __init__(self, map_size, curvature_points, validator_bb):
    """
    Args:
      map_size (int):          Map size in pixels (total map size
                               map_size*map_size).
      curvature_points (int):  How many road points are generated.
      validator_bb (function): A black box function which takes a test as an
                               input and returns 0 (invalid) or 1 (valid).
    """

    if map_size <= 0:
      raise ValueError("The map size must be positive.")
    if curvature_points < 2:
      raise ValueError("The roads must have at least two points.")

    super().__init__(map_size)

    self.ndimensions = curvature_points
    self.validator_bb = validator_bb
    self.target = 1

  def execute_test(self, tests):
    """
    Execute the given tests on the SUT.

    Args:
      tests (np.ndarray): Array of N tests with shape (N, self.ndimensions).

    Returns:
      result (np.ndarray): Array of shape (N, 1).
    """

    if len(tests.shape) != 2 or tests.shape[1] != self.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.ndimensions))

    result = np.zeros(shape=(tests.shape[0], 1))
    for n, test in enumerate(tests):
      result[n,0] = self.validator_bb.validity(test.reshape(1, test.shape[0]))[0,0]

    return result

  def sample_input_space(self, N=1):
    """
    Return n samples (tests) from the input space.

    Args:
      N (int): The number of tests to be sampled.

    Returns:
      tests (np.ndarray): Array of shape (N, curvature_points).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    return self._sample_input_space(N, self.ndimensions)

def sbst_test_to_image(test, sut):
  """
  Visualizes the road (described as points in the plane).
  """

  little_triangle = Polygon([(10, 0), (0, -5), (0, 5), (10, 0)])
  square = Polygon([(5, 5), (5, -5), (-5, -5), (-5, 5), (5, 5)])

  # TODO: describe arguments
  V = TestValidator(map_size=sut.map_size)
  try:
    the_test = RoadTestFactory.create_road_test(sut.test_to_road_points(test))
    valid, msg = V.validate_test(the_test)
  except:
    return

  # This code is mainly from https://github.com/se2p/tool-competition-av/code_pipeline/visualization.py
  plt.figure()

  # plt.gcf().set_title("Last Generated Test")
  plt.gca().set_aspect('equal', 'box')
  plt.gca().set(xlim=(-30, sut.map_size + 30), ylim=(-30, sut.map_size + 30))

  # Add information about the test validity
  title_string = "Test is " + ("valid" if valid else "invalid")
  if not valid:
    title_string = title_string + ":" + msg

  plt.suptitle(title_string, fontsize=14)
  plt.draw()

  # Plot the map. Trying to re-use an artist in more than one Axes which is supported
  map_patch = patches.Rectangle((0, 0), sut.map_size, sut.map_size, linewidth=1, edgecolor='black',
                                facecolor='none')
  plt.gca().add_patch(map_patch)

  # Road Geometry.
  road_poly = LineString([(t[0], t[1]) for t in the_test.interpolated_points]).buffer(8.0, cap_style=2, join_style=2)
  road_patch = PolygonPatch(road_poly, fc='gray', ec='dimgray')  # ec='#555555', alpha=0.5, zorder=4)
  plt.gca().add_patch(road_patch)

  # Interpolated Points
  sx = [t[0] for t in the_test.interpolated_points]
  sy = [t[1] for t in the_test.interpolated_points]
  plt.plot(sx, sy, 'yellow')

  # Road Points
  x = [t[0] for t in the_test.road_points]
  y = [t[1] for t in the_test.road_points]
  plt.plot(x, y, 'wo')

  # Plot the little triangle indicating the starting position of the ego-vehicle
  delta_x = sx[1] - sx[0]
  delta_y = sy[1] - sy[0]

  current_angle = atan2(delta_y, delta_x)

  rotation_angle = degrees(current_angle)
  transformed_fov = rotate(little_triangle, origin=(0, 0), angle=rotation_angle)
  transformed_fov = translate(transformed_fov, xoff=sx[0], yoff=sy[0])
  plt.plot(*transformed_fov.exterior.xy, color='black')

  # Plot the little square indicating the ending position of the ego-vehicle
  delta_x = sx[-1] - sx[-2]
  delta_y = sy[-1] - sy[-2]

  current_angle = atan2(delta_y, delta_x)

  rotation_angle = degrees(current_angle)
  transformed_fov = rotate(square, origin=(0, 0), angle=rotation_angle)
  transformed_fov = translate(transformed_fov, xoff=sx[-1], yoff=sy[-1])
  plt.plot(*transformed_fov.exterior.xy, color='black')

  plt.suptitle(title_string, fontsize=14)
  plt.draw()

  return plt.gcf()

def sbst_validate_test(test, sut):
  """
  Tests if the road (described as points in the plane) is valid.
  """

  # TODO: describe arguments

  V = TestValidator(map_size=sut.map_size)
  # Sometimes strange errors occur, and we work around them by declaring the
  # test as invalid.
  try:
    the_test = RoadTestFactory.create_road_test(sut.test_to_road_points(test))
    valid, msg = V.validate_test(the_test)
  except ValueError as e:
    if e.args[0] == "GEOSGeom_createLinearRing_r returned a NULL pointer":
      return 0
    raise

  #print(msg)
  return 1 if valid else 0

