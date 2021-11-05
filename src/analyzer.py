#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import torch

from sklearn.ensemble import AdaBoostRegressor

import neural_networks.wgan.analyzer

class Analyzer:
  """
  Base class for analyzers.
  """

  def __init__(self, input_dimension, device, logger=None):
    self.ndimensions = input_dimension
    self.device = device
    self.logger = logger
    self.log = lambda t: logger.log(t) if logger is not None else None

  def train_with_batch(self, dataX, dataY, train_settings, log=False):
    raise NotImplementedError()

  def save(self, identifier, path):
    raise NotImplementedError()

  def load(self, identifier, path):
    raise NotImplementedError()

  def predict(self, test):
    raise NotImplementedError()

class Analyzer_NN(Analyzer):
  """
  Analyzer based on a neural network.
  """

  def __init__(self, input_dimension, device, logger=None):
    # TODO: describe the arguments
    super().__init__(input_dimension, device, logger)

    # Number of neurons per layer in the neural networks.
    self.neurons = 128

    # How many bins are used for inverse frequency weighting.
    self.bins = 10

    self.modelA = neural_networks.wgan.analyzer.AnalyzerNetwork(input_shape=self.ndimensions, neurons=self.neurons).to(self.device)

    lr_a = 0.01
    self.optimizerA = torch.optim.Adam(self.modelA.parameters(), lr=lr_a, betas=(0, 0.9))

  def save(self, identifier, path):
    """
    Save the analyzer to the given path. File analyzer_{identifier} is created
    in the directory.
    """

    torch.save(self.modelA.state_dict(), os.path.join(path, "analyzer_{}".format(identifier)))

  def load(self, identifier, path):
    """
    Load the analyzer from path. File analyzer_{identifier} is expected to
    exist.
    """

    a_file_name = os.path.join(path, "analyzer_{}".format(identifier))

    if not os.path.exists(a_file_name):
      raise Exception("File '{}' does not exist in {}.".format(a_file_name, path))

    self.modelA.load_state_dict(torch.load(a_file_name))
    self.modelA.eval()

  def get_bin(self, x):
    """
    Return the bin of the number x in [0, 1].
    """

    i = int(x*self.bins)
    if i == self.bins: i -= 1
    return i

  def weights(self, data_Y):
    """
    Computes the bin weights needed for loss computation for the training data
    data_X, data_Y.
    """

    # We compute frequency of samples for each bin. The weights are the logs of
    # inverse frequencies.
    bin_freq = torch.zeros(self.bins)
    for n in range(data_Y.shape[0]):
      bin_freq[self.get_bin(data_Y[n,0])] += 1
    # If frequency is 0, we set it to 0.01.
    bin_freq[bin_freq == 0.0] = 0.01
    bin_freq = bin_freq / data_Y.shape[0]
    bin_freq.pow_(-1).log_()

    return bin_freq

  def analyzer_loss(self, data_X, data_Y, weights):
    """
    Computes the analyzer loss for data_X given real outputs data_Y according
    to the bin weights of the model training data.
    """

    # Compute the weight vector for the input.
    # TODO: Is there a more efficient way to compute this?
    weight_vector = torch.zeros(size=(data_Y.shape[0], 1)).to(self.device)
    for n in range(weight_vector.shape[0]):
      weight_vector[n,0] = weights[self.get_bin(data_Y[n,0])]

    # We map the values from [0, 1] to \R using a logit transformation so that
    # weighted MSE loss works better. Since logit is undefined in 0 and 1, we
    # actually first transform the values to the interval [0.01, 0.99].
    #model_loss = (torch.abs(self.modelA(data_X) - data_Y)).mean()
    #model_loss = (weight_vector*torch.abs(self.modelA(data_X) - data_Y)).mean()
    model_loss = ((self.modelA(data_X) - data_Y)**2).mean()
    #model_loss = (weight_vector*(self.modelA(data_X) - data_Y)**2).mean()
    #model_loss = ((torch.logit(0.98*self.modelA(data_X) + 0.01) - torch.logit(0.98*data_Y + 0.01))**2).mean()
    #model_loss = (weight_vector*(torch.logit(0.98*self.modelA(data_X) + 0.01) - torch.logit(0.98*data_Y + 0.01))**2).mean()

    # Compute L2 regularization.
    l2_regularization = 0
    for parameter in self.modelA.parameters():
      l2_regularization += torch.sum(torch.square(parameter))

    # TODO: make configurable
    A_loss = model_loss + 0.01*l2_regularization

    return A_loss

  def train_with_batch(self, data_X, data_Y, epoch_settings, log=False):
    """
    Train the analyzer part of the model with a batch of training data.

    Args:
      data_X (np.ndarray):   Array of tests of shape (N, self.sut.ndimensions).
      data_Y (np.ndarray):   Array of test outputs of shape (N, 1).
      epoch_settings (dict): A dictionary setting up the number of training
                             epochs for various parts of the model. The keys
                             are as follows:

                               analyzer_epochs: How many total runs are made
                               with the given training data.

                             The default for each missing key is 1. Keys not
                             found above are ignored.
      log (bool):            Log additional information on epochs and losses.
    """

    if len(data_X.shape) != 2 or data_X.shape[1] != self.ndimensions:
      raise ValueError("Array data_X expected to have shape (N, {}).".format(self.ndimensions))
    if len(data_Y.shape) != 2 or data_Y.shape[0] < data_X.shape[0]:
      raise ValueError("Array data_Y array should have at least as many elements as there are tests.")

    data_X = torch.from_numpy(data_X).float().to(self.device)
    data_Y = torch.from_numpy(data_Y).float().to(self.device)

    # Unpack values from the epochs dictionary.
    analyzer_epochs = epoch_settings["analyzer_epochs"] if "analyzer_epochs" in epoch_settings else 1

    # Save the training modes for later restoring.
    training_A = self.modelA.training

    # Find the weights for the training data.
    weights = self.weights(data_Y)

    # Train the analyzer.
    # -----------------------------------------------------------------------
    self.modelA.train(True)
    for m in range(analyzer_epochs):
      A_loss = self.analyzer_loss(data_X, data_Y, weights)
      self.optimizerA.zero_grad()
      A_loss.backward()
      self.optimizerA.step()

      if log:
        self.log("Analyzer epoch {}/{}, Loss: {}".format(m + 1, analyzer_epochs, A_loss))

    # Visualize the computational graph.
    #print(make_dot(A_loss, params=dict(self.modelA.named_parameters())))

    self.modelA.train(training_A)

  def predict(self, test):
    """
    Predicts the fitness of the given test.

    Args:
      test (np.ndarray): Array of shape (N, self.ndimensions).

    Returns:
      output (np.ndarray): Array of shape (N, 1).
    """

    if len(test.shape) != 2 or test.shape[1] != self.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.ndimensions))

    test_tensor = torch.from_numpy(test).float().to(self.device)
    return self.modelA(test_tensor).cpu().detach().numpy()

class Analyzer_AdaBoost(Analyzer):
  """
  Analyzer based on AdaBoost.
  """

  def __init__(self, input_dimension, device, logger=None):
    # TODO: describe the arguments
    super().__init__(input_dimension, device, logger)

    self.modelA = AdaBoostRegressor(n_estimators=10, loss="linear")

  def train_with_batch(self, data_X, data_Y, epoch_settings, log=False):
    """
    Train the analyzer part of the model with a batch of training data.

    Args:
      data_X (np.ndarray):   Array of tests of shape (N, self.sut.ndimensions).
      data_Y (np.ndarray):   Array of test outputs of shape (N, 1).
      epoch_settings (dict): A dictionary setting up the number of training
                             epochs for various parts of the model. The keys
                             are as follows:

                               analyzer_epochs: How many total runs are made
                               with the given training data.

                             The default for each missing key is 1. Keys not
                             found above are ignored.
      log (bool):            Log additional information on epochs and losses.
    """

    self.modelA.fit(data_X, data_Y.reshape(data_Y.shape[0]))

  def predict(self, test):
    """
    Predicts the fitness of the given test.

    Args:
      test (np.ndarray): Array of shape (N, self.ndimensions).

    Returns:
      output (np.ndarray): Array of shape (N, 1).
    """

    if len(test.shape) != 2 or test.shape[1] != self.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.ndimensions))

    return self.modelA.predict(test).reshape(test.shape[0], 1)
