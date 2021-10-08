#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.model_selection import train_test_split
import numpy as np

from config import *
from sut.sut import SUT

class OdroidSUT(SUT):
  """
  Implements the Odroid system under test.
  """

  def __init__(self, output, target):
    """
    Initialize the class.

    Args:
      output (int):   Which output data is used (1 = power, 2 = performance,
                      3 = efficiency).
      target (float): Threshold for a positive test value.
    """

    if not (1 <= output <= 3):
      raise Exception("Argument 'output' should be 1, 2 or 3.")

    super().__init__()
    self.output = output
    self.target = target

    self.ndimensions = None
    self.dataX = None
    self.dataY = None
    self.scaleX = None
    self.scaleY = None

    try:
      self._load_odroid_data(output, target)
    except:
      raise

  def _load_odroid_data(self, output, target):
    if not (1 <= output <= 3):
      raise Exception("Argument 'output' should be 1, 2 or 3.")

    file_name = config["odroid_file_base"] + ".npy"

    if not os.path.exists(file_name):
      preprocess_data.generate_odroid_data()

    data = np.load(file_name)

    # Set number of input dimensions.
    self.ndimensions = 6

    self.dataX = data[: , 0:self.ndimensions]
    self.dataY = data[: , self.ndimensions + output - 1]

    # Compute the number of positive tests in the data.
    self.total = self.dataY.shape[0]
    self.totalp = self.dataY[self.dataY >= self.target].shape[0]

    # Normalize the input to [-1, 1].
    self.scaleX = self.dataX.max(axis=0)
    self.dataX = (self.dataX / self.scaleX)*2 - 1
    # Normalize the output to [0, 1] (in order to use certain loss functions).
    self.scaleY = self.dataY.max(axis=0)
    self.dataY = self.dataY / self.scaleY
    # Scale the positive threshold as well.
    self.target = self.target / self.scaleY

  def execute_test(self, tests):
    """
    Execute the given tests on the SUT. As not all possible parameters have a
    test value in the data, we find the closest test from the test data
    (Euclidean distance) and return its value.

    Args:
      tests (np.ndarray): Array of N tests with shape (N, self.ndimensions).

    Returns:
      result (np.ndarray): Array of shape (N, 1).
    """

    if len(tests.shape) != 2 or tests.shape[1] != self.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.ndimensions))

    result = np.zeros(shape=(tests.shape[0], 1))
    for n in range(tests.shape[0]):
      distances = np.sum((self.dataX - tests[n,:])**2, axis=1)
      result[n,0] = self.dataY[np.argmin(distances)]

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

    # We simply sample n tests from the data and return the corresponding
    # outputs.
    # TODO: what to do if n > self.dataX.shape[0]?
    sampleX, _, sampleY, _ = train_test_split(self.dataX, self.dataY,
                                              train_size=N, test_size=1)
    return sampleX, sampleY.reshape(sampleX.shape[0], 1)

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

    # We simply sample n tests from the data.
    # TODO: what to do if n > self.dataX.shape[0]?
    sampleX, _, _, _ = train_test_split(self.dataX, self.dataY, train_size=N,
                                        test_size=1)
    return sampleX

def generate_odroid_data():
  """
  This function loads the Odroid data from CSV, converts it to appropriate
  form, and saves it to a Numpy .npy file.
  """

  """
  Data format:
  <configuration name>,<power>,<performance>,<efficiency>

  The configuration name is a string with 5 parts separated by / which
  correspond to input parameters. An example is

  4a7/1100Mhz/100%3a15/800Mhz/100%

  The parts 2,4,5 are used as such by removing 'Mhz' and '%'. The first part is
  mapped to an integer in order of appearance, and so is the segment of the
  third part after '%'. This gives a total 6 input parameters. The above line
  maps to

  0,1100,100,1,800,100
  """

  # This function returns encodings of strings to integers.
  encoding = {}
  def encode(s):
    if not s in encoding:
      if len(encoding) == 0:
        encoding[s] = 0
      else:
        encoding[s] = max(encoding.values()) + 1

    return encoding[s]

  data = []
  with open(config["odroid_file_base"] + ".csv", mode="r") as f:
    c = 0
    skip = 1
    while True:
      line = f.readline()
      if line == "": break
      c += 1
      if c <= skip: continue

      pcs = line.split(",")
      w = pcs[0].split("/")
      if len(w) < 5:
        print("Line {} malformed: {}".format(c, line))
        continue

      new = []
      # Test input variables.
      new.append(encode(w[0]))
      new.append(int(w[1][:-3]))
      w2 = w[2].split("%")
      new.append(int(w2[0]))
      new.append(encode(w2[1]))
      new.append(int(w[3][:-3]))
      new.append(int(w[4][:-1]))
      # Test outputs.
      new.append(float(pcs[1]))
      new.append(float(pcs[2]))
      new.append(float(pcs[3]))

      data.append(new)

  np.save(config["odroid_file_base"] + ".npy", data)

