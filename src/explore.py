#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, os, json

import matplotlib.pyplot as plt
import numpy as np

from code_pipeline.tests_generation import RoadTestFactory

from config import *
from logger import Logger
from session import Session
from sut.sut_sbst import *

def closest(t, X, k=1):
  """
  Finds k tests from X which are closest to t.
  """

  if len(X) == 0:
    return []

  M = model.sut.test_to_road_points
  D = frechet_distance

  t = M(t)
  distances = [D(t, M(X[0,:]))]
  idx = [0]

  for n in range(1, X.shape[0]):
    d = D(t, M(X[n,:]))
    for i in range(len(distances) - 1, -1, -1):
      if distances[i] < d:
        distances.insert(i+1, d)
        idx.insert(i+1, n)
        break
    if distances[i] > d:
      distances.insert(0, d)
      idx.insert(0, n)
    distances = distances[:k]
    idx = idx[:k]

  return idx

def draw_test(test, ax, color="b"):
  """
  Draw the given test to the specified axis.
  """

  the_test = RoadTestFactory.create_road_test(model.sut.test_to_road_points(test))
  road_points = [(r[0], r[1]) for r in the_test.interpolated_points]

  p_x = [r[0] for r in road_points]
  p_y = [r[1] for r in road_points]
  ax.plot(p_x, p_y, color)

  x = [t[0] for t in the_test.road_points]
  y = [t[1] for t in the_test.road_points]
  ax.plot(x, y, "{}o".format(color))

def draw_tests(tests, outputs, colors, closest_tests, closest_labels, height, width):
  fig, axes = plt.subplots(height, width, sharex=True, sharey=True, figsize=(50, 50))

  n = 0
  for i in range(height):
    for j in range(width):
      ax = axes[i, j]
      # Find the closest test from set of closest tests.
      if len(closest_labels) > 0:
        idx = closest(tests[n, :], closest_tests, k=3)
        idx = idx[np.argmax(outputs[idx])]
        d = round(frechet_distance(model.sut.test_to_road_points(tests[n, :]),
                                   model.sut.test_to_road_points(closest_tests[idx, :])), 2)
        label = closest_labels[idx]
      else:
        label = ""
        d = -1
      ft = str(round(outputs[n, 0], 2))
      ax.set_title("{}, cl: {} ({}), ft: {}".format(n + 1, label, d, ft))
      color = colors[n] if n in colors else "b"
      draw_test(tests[n], ax, color)
      n += 1
      if n == tests.shape[0]: break
    if n == tests.shape[0]: break

  return fig

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print("The command line arguments should specify sut_id, model_id, and a session directory.")
    raise SystemExit

  sut_id = sys.argv[1]
  if not sut_id in config["available_sut"]:
    raise ValueError("The sut_id '{}' is invalid.".format(sut_id))

  model_id = sys.argv[2]
  if not model_id in config["available_model"]:
    raise ValueError("The model_id '{}' is invalid.".format(model_id))

  enable_log_printout = True

  logger = Logger(quiet=not enable_log_printout)
  model, _view_test, _save_test = get_model(sut_id, model_id, logger)

  session_directory = sys.argv[3]
  session = os.path.basename(os.path.normpath(session_directory))

  session = Session(model_id, sut_id, session)
  session.add_saved_parameter(*config["session_attributes"][model_id])
  session.session_directory = sys.argv[3]
  session.load()

  # Load the session training data.
  with open(os.path.join(session.session_directory, "training_data.npy"), mode="rb") as f:
    X = np.load(f)
    Y = np.load(f)

  # Which test round is selected.
  K = 300

  colors = {n:"r" if Y[n,0] >= model.sut.target else "b" for n in range(K)}

  fig = draw_tests(X[:K], Y[:K], colors, X[:50], list(range(1, 51)), 30, 10)
  fig.savefig(os.path.join(session.session_directory, "test_visualization.png"))
  raise SystemExit

  # Load a model snapshot.
  #model.load("init", session.session_directory)
  model.load("model_snapshot_{}".format(K), session.session_directory)

  # Generate some tests and draw them.
  new_tests = model.generate_test(20)
  fig = draw_tests(new_tests, model.analyzer.predict(new_tests), {}, X, list(range(1, len(X) + 1)), 5, 4)
  fig.savefig(os.path.join(session.session_directory, "generator_output.png"))

  #plt.show()
  #plt.gcf().savefig("whatever_{:>3}.png".format(N))
