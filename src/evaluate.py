#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys

from config import config, get_model

if __name__ == "__main__":
  if len(sys.argv) < 4:
    print("The command line arguments should specify sut_id, model_id, and a session directory.")
    raise SystemExit

  sut_id = sys.argv[1]
  model_id = sys.argv[2]
  session_directory = sys.argv[3]

  if not os.path.exists(session_directory):
    print("Directory '{}' does not exist.".format(session_directory))

  model, _view_test, _save_test = get_model(sut_id, model_id)
  model.load(session_directory)

  # Generate new samples to assess quality visually.
  for n, test in enumerate(model.generate_test(30)):
    view_test(test)
    save_test(test, "eval_{}".format(n + 1))

