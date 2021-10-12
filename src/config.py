#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os

config = {}
config["data_path"] = os.path.join("..", "data")

config["odroid"] = {"file_base": os.path.join(config["data_path"], "odroid", "odroid")}

config["sbst"] = {"data_directory": os.path.join(config["data_path"], "sbst")}
config["sbst"]["beamng_home"] = "C:\\Users\\japel\\dev\\BeamNG"
config["sbst"]["validator_training_data"] = os.path.join(config["sbst"]["data_directory"], "validator_training_data.npy")
config["sbst"]["validator_neural_network"] = os.path.join(config["sbst"]["data_directory"], "validator_neural_network")
config["sbst"]["pregenerated_initial_data"] = os.path.join(config["sbst"]["data_directory"], "pregenerated_initial_data.npy")

