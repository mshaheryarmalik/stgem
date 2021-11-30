#!/usr/bin/python3
# -*- coding: utf-8 -*-

import sys, time

import numpy as np

from config import *

if __name__ == "__main__":
  sys.argv.append("sbst")
  sys.argv.append("[-0.7432606,  -0.18621023, -0.95667195, -0.9563363,  -0.97654086]")
  if len(sys.argv) < 3:
    print("The command line arguments should specify sut_id and a test.")
    raise SystemExit

  sut_id = sys.argv[1]
  if not sut_id in config["available_sut"]:
    raise ValueError("The sut_id '{}' is invalid.".format(sut_id))

  test = sys.argv[2].strip()
  if test.startswith("["):
    test = test[1:-1].strip()
    if test.startswith("["):
      test = test[1:-1].strip()
    if not "," in test:
      s = ""
      emitted = False
      for c in test:
        if c == " " and not emitted:
          s += ","
          emitted = True
        elif c != " ":
          s += c
      test = s
  test = np.array([float(x) for x in test.split(",")]).reshape(1, -1)

  print("Executing test {}.".format(test))

  model, _view_test, _save_test = get_model(sut_id, model_id, logger=None)
  start = time.monotonic()
  output = model.sut.execute_test(test)
  end = time.monotonic()

  print("Test output: {}".format(output))
  print("Execution time: {} s.".format(round(end - start, 2)))
