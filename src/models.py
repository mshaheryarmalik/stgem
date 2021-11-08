#!/usr/bin/python3
# -*- coding: utf-8 -*-

import os

import numpy as np

from sklearn.ensemble import AdaBoostRegressor

import torch
import torch.nn as nn

# For visualizing the computational graphs.
#from torchviz import make_dot

from analyzer import *

import neural_networks.ogan.generator
import neural_networks.ogan.discriminator

import neural_networks.wgan.generator
import neural_networks.wgan.critic

class Model:
  """
  Base class for all models.
  """

  def __init__(self, sut, validator, device, logger=None):
    self.sut = sut
    self.device = device
    self.validator = validator
    self.logger = logger
    self.log = lambda t: logger.log(t) if logger is not None else None

    # Settings for training. These are set externally.
    self.epoch_settings_init = None
    self.epoch_settings = None
    self.random_init = None
    self.N_tests = None

  @property
  def parameters(self):
    return {k:getattr(self, k) for k in ["random_init", "N_tests", "epoch_settings", "epoch_settings_init"]}

  def train_with_batch(self, dataX, dataY, epoch_settings, log=False):
    raise NotImplementedError()

  def generate_test(self, N=1):
    raise NotImplementedError()

  def save(self, path):
    raise NotImplementedError()

  def load(self, path):
    raise NotImplementedError()

  def validity(self, tests):
    """
    Validate the given test using the true validator.

    Args:
      tests (np.ndarray): Array of shape (N, self.sut.ndimensions).

    Returns:
      output (np.ndarray): Array of shape (N, 1).
    """

    if len(tests.shape) != 2 or tests.shape[1] != self.sut.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.sut.ndimensions))

    if self.validator is None:
      result = np.ones(shape=(tests.shape[0], 1))
    else:
      result = self.validator.validity(tests)

    return result

class OGAN(Model):
  """
  Implements the OGAN model.
  """

  def __init__(self, sut, validator, device, logger=None):
    # TODO: describe the arguments
    super().__init__(sut, validator, device, logger)

    self.modelG = None
    self.modelD = None
    # Input dimension for the noise inputted to the generator.
    self.noise_dim = 100
    # Number of neurons per layer in the neural networks.
    self.neurons = 128

    # Initialize neural network models.
    self.modelG = neural_networks.ogan.generator.GeneratorNetwork(input_shape=self.noise_dim, output_shape=self.sut.ndimensions, neurons=self.neurons).to(self.device)
    self.modelD = neural_networks.ogan.discriminator.DiscriminatorNetwork(input_shape=self.sut.ndimensions, neurons=self.neurons).to(self.device)

    # Loss functions.
    # TODO: figure out a reasonable default and make configurable.
    self.lossG = nn.MSELoss()
    #self.lossG = nn.BCELoss() # binary cross entropy
    #self.lossD = nn.L1Loss()
    self.lossD = nn.MSELoss() # mean square error

    # Optimizers.
    # TODO: figure out reasonable defaults and make configurable.
    lr = 0.001
    #self.optimizerD = torch.optim.RMSprop(self.modelD.parameters(), lr=lr)
    #self.optimizerG = torch.optim.RMSprop(self.modelG.parameters(), lr=lr)
    self.optimizerD = torch.optim.Adam(self.modelD.parameters(), lr=lr)
    self.optimizerG = torch.optim.Adam(self.modelG.parameters(), lr=lr)

  def save(self, identifier, path):
    """
    Save the model to the given path. Files discriminator and generator are
    created in the directory.
    """

    torch.save(self.modelD.state_dict(), os.path.join(path, "discriminator_{}".format(identifier)))
    torch.save(self.modelG.state_dict(), os.path.join(path, "generator_{}".format(identifier)))

  def load(self, identifier, path):
    """
    Load the model from path. Files discriminator and analyzer are expected to
    exist.
    """

    d_file_name = os.path.join(path, "discriminator_{}".format(identifier))
    g_file_name = os.path.join(path, "generator_{}".format(identifier))

    if not os.path.exists(d_file_name):
      raise Exception("File '{}' does not exist in {}.".format(d_file_name, path))
    if not os.path.exists(os.path.join(path, "generator")):
      raise Exception("File '{}' does not exist in {}.".format(g_file_name, path))

    self.modelD.load_state_dict(torch.load(d_file_name))
    self.modelG.load_state_dict(torch.load(g_file_name))
    self.modelD.eval()
    self.modelG.eval()

  def train_with_batch(self, dataX, dataY, epoch_settings, log=False):
    """
    Train the OGAN with a batch of training data.

    Args:
      dataX (np.ndarray):             Array of tests of shape
                                      (N, self.sut.ndimensions).
      dataY (np.ndarray):             Array of test outputs of shape (N, 1).
      epoch_settings (dict): A dictionary setting up the number of training
                             epochs for various parts of the model. The keys
                             are as follows:

                               epochs: How many total runs are made with the
                               given training data.

                               discriminator_epochs: How many times the
                               discriminator is trained per epoch.

                               generator_epochs: How many times the generator
                               is trained per epoch.

                             The default for each missing key is 1. Keys not
                             found above are ignored.
      log (bool):            Log additional information on epochs and losses.
    """

    if len(dataX.shape) != 2 or dataX.shape[1] != self.sut.ndimensions:
      raise ValueError("Test array expected to have shape (N, {}).".format(self.ndimensions))
    if len(dataY.shape) != 2 or dataY.shape[0] < dataX.shape[0]:
      raise ValueError("Output array should have at least as many elements as there are tests.")

    dataX = torch.from_numpy(dataX).float().to(self.device)
    dataY = torch.from_numpy(dataY).float().to(self.device)

    # Unpack values from the epochs dictionary.
    epochs = epoch_settings["epochs"] if "epochs" in epoch_settings else 1
    discriminator_epochs = epoch_settings["discriminator_epochs"] if "discriminator_epochs" in epoch_settings else 1
    generator_epochs = epoch_settings["generator_epochs"] if "generator_epochs" in epoch_settings else 1

    # Save the training modes for later restoring.
    training_D = self.modelD.training
    training_G = self.modelG.training
 
    for n in range(epochs):
      # Train the discriminator.
      # -----------------------------------------------------------------------
      # We want the discriminator to learn the mapping from tests to test
      # outputs.
      self.modelD.train(True)
      for m in range(discriminator_epochs):
        # We the values from [0, 1] to \R using a logit transformation so that
        # MSE loss works better. Since logit is undefined in 0 and 1, we
        # actually first transform the values to the interval [0.01, 0.99].
        D_loss = self.lossD(torch.logit(0.98*self.modelD(dataX) + 0.01), torch.logit(0.98*dataY + 0.01))
        self.optimizerD.zero_grad()
        D_loss.backward()
        self.optimizerD.step()

        if log:
          self.log("Epoch {}/{}, Discriminator epoch {}/{}, Loss: {}".format(n + 1, epochs, m + 1, discriminator_epochs, D_loss))

      self.modelD.train(False)

      # Visualize the computational graph.
      #print(make_dot(D_loss, params=dict(self.modelD.named_parameters())))

      # Train the generator on the discriminator.
      # -----------------------------------------------------------------------
      # We generate noise and label it to have output 1 (high fitness).
      # Training the generator in this way should shift it to generate tests
      # with high output values (high fitness).
      if self.validator is not None:
        # We need to generate valid tests in order not to confuse the generator
        # by garbage inputs (invalid tests with high fitness do not exist).
        # TODO: Is the following line really needed when not using batch
        #       normalization?
        self.modelG.train(False)
        # TODO: put size into parameter
        inputs = np.zeros(shape=(10, self.sut.ndimensions))
        k = 0
        while k < discriminator_data_size:
          new_test = self.modelG(((torch.rand(size=(1, self.modelG.input_shape)) - 0.5)/0.5).to(self.device)).detach().numpy()
          if self.validator.validity(new_test)[0,0] == 0.0: continue
          inputs[k,:] = new_test[0,:]
          k += 1

        self.modelG.train(True)
        inputs = torch.from_numpy(inputs).float.to(self.device)
      else:
        inputs = ((torch.rand(size=(10, self.modelG.input_shape)) - 0.5)/0.5).to(self.device)

      fake_label = torch.ones(size=(10, 1)).to(self.device)

      # Notice the following subtlety. Above the tensor 'outputs' contains
      # information on how it is computed (the computation graph is being kept
      # track off) up to the original input 'noise' which does not anymore
      # depend on previous operations. Since 'self.modelD' is used as part of
      # the computation, its parameters are present in the computation graph.
      # These parameters are however not updated because the optimizer is
      # initialized only for the parameters of 'self.modelG' (see the
      # initialization of 'self.modelG'.

      for k in range(generator_epochs):
        outputs = self.modelD(self.modelG(inputs))
        # Same comment as above on D_loss.
        G_loss = self.lossG(torch.logit(0.98*outputs + 0.01), torch.logit(0.98*fake_label + 0.01))
        self.optimizerG.zero_grad()
        G_loss.backward()
        self.optimizerG.step()
        if log:
          self.log("Epoch {}/{}, Generator epoch: {}/{}, Loss: {}".format(n + 1, epochs, k + 1, generator_epochs, G_loss))

      self.modelG.train(False)

      # Visualize the computational graph.
      #print(make_dot(G_loss, params=dict(self.modelG.named_parameters())))

    # Restore the training modes.
    self.modelD.train(training_D)
    self.modelG.train(training_G)

  def generate_test(self, N=1):
    """
    Generate N random tests.

    Args:
      N (int): Number of tests to be generated.

    Returns:
      output (np.ndarray): Array of shape (N, self.sut.ndimensions).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    # Generate uniform noise in [-1, 1].
    noise = ((torch.rand(size=(N, self.noise_dim)) - 0.5)/0.5).to(self.device)
    return self.modelG(noise).cpu().detach().numpy()

  def predict_fitness(self, test):
    """
    Predicts the fitness of the given test.

    Args:
      test (np.ndarray): Array of shape (N, self.sut.ndimensions).

    Returns:
      output (np.ndarray): Array of shape (N, 1).
    """

    if len(test.shape) != 2 or test.shape[1] != self.sut.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.sut.ndimensions))

    test_tensor = torch.from_numpy(test).float().to(self.device)
    return self.modelD(test_tensor).cpu().detach().numpy()

class WGAN(Model):
  """
  Implements the WGAN model.
  """

  def __init__(self, sut, validator, device, logger=None):
    # TODO: describe the arguments
    super().__init__(sut, validator, device, logger)

    # The coefficient for the loss gradient penalty term.
    self.gp_coefficient = None

    self.modelG = None
    self.modelC = None
    self.analyzer = None
    # Input dimension for the noise inputted to the generator.
    self.noise_dim = 100
    # Number of neurons per layer in the neural networks.
    self.neurons = 128

    # Initialize neural network models.
    self.modelG = neural_networks.wgan.generator.GeneratorNetwork(input_shape=self.noise_dim, output_shape=self.sut.ndimensions, neurons=self.neurons).to(self.device)
    self.modelC = neural_networks.wgan.critic.CriticNetwork(input_shape=self.sut.ndimensions, neurons=self.neurons).to(self.device)
    self.analyzer = Analyzer_NN(self.sut.ndimensions, self.device, self.logger)
    #self.analyzer = Analyzer_RandomForest(self.sut.ndimensions, self.device, self.logger)

    # Optimizers.
    # TODO: figure out reasonable defaults and make configurable.
    lr_wgan = 0.00005
    #self.optimizerG = torch.optim.RMSprop(self.modelG.parameters(), lr=lr_wgan) # RMSprop with clipping
    #self.optimizerC = torch.optim.RMSprop(self.modelC.parameters(), lr=lr_wgan) # RMSprop with clipping
    self.optimizerG = torch.optim.Adam(self.modelG.parameters(), lr=lr_wgan, betas=(0, 0.9))
    self.optimizerC = torch.optim.Adam(self.modelC.parameters(), lr=lr_wgan, betas=(0, 0.9))

  def save(self, identifier, path):
    """
    Save the model to the given path. Files critic_{identifier},
    generator_{identifier}, and analyzer_{identifier} are created in the
    directory.
    """

    torch.save(self.modelC.state_dict(), os.path.join(path, "critic_{}".format(identifier)))
    torch.save(self.modelG.state_dict(), os.path.join(path, "generator_{}".format(identifier)))
    self.analyzer.save(identifier, path)

  def load(self, identifier, path):
    """
    Load the model from path. Files critic_{identifier},
    generator_{identifier}, and analyzer_{identifier} are expected to exist.
    """

    c_file_name = os.path.join(path, "critic_{}".format(identifier))
    g_file_name = os.path.join(path, "generator_{}".format(identifier))

    if not os.path.exists(c_file_name):
      raise Exception("File '{}' does not exist in {}.".format(c_file_name, path))
    if not os.path.exists(g_file_name):
      raise Exception("File '{}' does not exist in {}.".format(g_file_name, path))

    self.modelC.load_state_dict(torch.load(c_file_name))
    self.modelG.load_state_dict(torch.load(g_file_name))
    self.modelC.eval()
    self.modelG.eval()
    self.analyzer.load(identifier, path)

  def train_analyzer_with_batch(self, data_X, data_Y, epoch_settings, log=False):
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

    self.analyzer.train_with_batch(data_X, data_Y, epoch_settings, log=log)

  def train_with_batch(self, data_X, data_Y, epoch_settings, log=False):
    """
    Train the WGAN with a batch of training data.

    Args:
      data_X (np.ndarray):   Array of tests of shape (M, self.sut.ndimensions).
      data_Y (np.ndarray):   Array of test outputs of shape (M, 1).
      epoch_settings (dict): A dictionary setting up the number of training
                             epochs for various parts of the model. The keys
                             are as follows:

                               epochs: How many total runs are made with the
                               given training data.

                               critic_epochs: How many times the critic is
                               trained per epoch.

                               generator_epochs: How many times the generator
                               is trained per epoch.

                             The default for each missing key is 1. Keys not
                             found above are ignored.
      log (bool):            Log additional information on epochs and losses.
    """

    if len(data_X.shape) != 2 or data_X.shape[1] != self.sut.ndimensions:
      raise ValueError("Array data_X expected to have shape (N, {}).".format(self.ndimensions))
    if len(data_Y.shape) != 2 or data_Y.shape[0] < data_X.shape[0]:
      raise ValueError("Array data_Y array should have at least as many elements as there are tests.")

    data_X = torch.from_numpy(data_X).float().to(self.device)
    data_Y = torch.from_numpy(data_Y).float().to(self.device)

    # Unpack values from the epochs dictionary.
    epochs = epoch_settings["epochs"] if "epochs" in epoch_settings else 1
    critic_epochs = epoch_settings["critic_epochs"] if "critic_epochs" in epoch_settings else 1
    generator_epochs = epoch_settings["generator_epochs"] if "generator_epochs" in epoch_settings else 1

    # Save the training modes for later restoring.
    training_C = self.modelC.training
    training_G = self.modelG.training

    for n in range(epochs):
      # Train the critic.
      # -----------------------------------------------------------------------
      self.modelC.train(True)
      for m in range(critic_epochs):
        # Here the mini batch size of the WGAN-GP is set to be the number of
        # training samples for the critic
        M = data_X.shape[0]

        # Loss on real data.
        real_inputs = data_X
        real_outputs = self.modelC(real_inputs)
        real_loss = real_outputs.mean(0)

        # Loss on generated data.
        # For now we use as much generated data as we have real data.
        noise = ((torch.rand(size=(M, self.modelG.input_shape)) - 0.5)/0.5).to(self.device)
        fake_inputs = self.modelG(noise)
        fake_outputs = self.modelC(fake_inputs)
        fake_loss = fake_outputs.mean(0)

        # Gradient penalty.
        # Compute interpolated data.
        e = torch.rand(size=(M, 1)).to(self.device)
        interpolated_inputs = e*real_inputs + (1-e)*fake_inputs
        # Get critic output on interpolated data.
        interpolated_outputs = self.modelC(interpolated_inputs)
        # Compute the gradients wrt to the interpolated inputs.
        # Warning: Showing the validity of the following line requires some pen
        #          and paper calculations.
        gradients = torch.autograd.grad(inputs=interpolated_inputs,
                                        outputs=interpolated_outputs,
                                        grad_outputs=torch.ones_like(interpolated_outputs).to(self.device),
                                        create_graph=True,
                                        retain_graph=True)[0]

        # We add epsilon for stability.
        epsilon = 0.000001
        gradients_norms = torch.sqrt(torch.sum(gradients**2, dim=1) + epsilon)
        gradient_penalty = gradients_norms.mean()
        #gradient_penalty = ((torch.linalg.norm(gradients, dim=1) - 1)**2).mean()

        C_loss = fake_loss - real_loss + self.gp_coefficient*gradient_penalty
        self.optimizerC.zero_grad()
        C_loss.backward()
        self.optimizerC.step()

        if log:
          self.log("Epoch {}/{}, Critic epoch {}/{}, Loss: {}, GP: {}".format(n + 1, epochs, m + 1, critic_epochs, C_loss[0], self.gp_coefficient*gradient_penalty))

      self.modelC.train(False)

      # Visualize the computational graph.
      #print(make_dot(C_loss, params=dict(self.modelC.named_parameters())))

      # Train the generator.
      # -----------------------------------------------------------------------
      self.modelG.train(True)
      for m in range(generator_epochs):
        # For now we use as much generated data as we have real data.
        noise = ((torch.rand(size=(data_X.shape[0], self.modelG.input_shape)) - 0.5)/0.5).to(self.device)
        outputs = self.modelC(self.modelG(noise))

        G_loss = -outputs.mean(0)
        self.optimizerG.zero_grad()
        G_loss.backward()
        self.optimizerG.step()

        if log:
          self.log("Epoch {}/{}, Generator epoch {}/{}, Loss: {}".format(n + 1, epochs, m + 1, generator_epochs, G_loss[0]))

      self.modelG.train(False)

      if log:
        # Same as above in critic training.
        real_inputs = data_X
        real_outputs = self.modelC(real_inputs)
        real_loss = real_outputs.mean(0)

        # For now we use as much generated data as we have real data.
        noise = ((torch.rand(size=(real_inputs.shape[0], self.modelG.input_shape)) - 0.5)/0.5).to(self.device)
        fake_inputs = self.modelG(noise)
        fake_outputs = self.modelC(fake_inputs)
        fake_loss = fake_outputs.mean(0)

        W_distance = real_loss - fake_loss

        self.log("Epoch {}/{}, W. distance: {}".format(n + 1, epochs, W_distance[0]))

      # Visualize the computational graph.
      #print(make_dot(G_loss, params=dict(self.modelG.named_parameters())))

    # Restore the training modes.
    self.modelC.train(training_C)
    self.modelG.train(training_G)

  def generate_test(self, N=1):
    """
    Generate N random tests.

    Args:
      N (int): Number of tests to be generated.

    Returns:
      output (np.ndarray): Array of shape (N, self.sut.ndimensions).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    training_G = self.modelG.training
    # Generate uniform noise in [-1, 1].
    noise = ((torch.rand(size=(N, self.noise_dim)) - 0.5)/0.5).to(self.device)
    self.modelG.train(False)
    result = self.modelG(noise).cpu().detach().numpy()
    self.modelG.train(training_G)
    return result

  def predict_fitness(self, test):
    """
    Predicts the fitness of the given test.

    Args:
      test (np.ndarray): Array of shape (N, self.sut.ndimensions).

    Returns:
      output (np.ndarray): Array of shape (N, 1).
    """

    return self.analyzer.predict(test)

class RandomGenerator(Model):
  """
  Implements the random test generator.
  """

  def __init__(self, sut, validator, device, logger=None):
    # TODO: describe the arguments
    super().__init__(sut, validator, device, logger)

  def train_with_batch(self, dataX, dataY, epoch_settings, log=False):
    pass

  def save(self, path):
    pass

  def load(self, path):
    pass

  def generate_test(self, N=1):
    """
    Generate N random tests.

    Args:
      N (int): Number of tests to be generated.

    Returns:
      output (np.ndarray): Array of shape (N, self.sut.ndimensions).
    """

    if N <= 0:
      raise ValueError("The number of tests should be positive.")

    return np.random.uniform(-1, 1, (N, self.sut.ndimensions))

  def predict_fitness(self, test):
    """
    Predicts the fitness of the given test.

    Args:
      test (np.ndarray): Array of shape (N, self.sut.ndimensions).

    Returns:
      output (np.ndarray): Array of shape (N, 1).
    """

    if len(test.shape) != 2 or test.shape[1] != self.sut.ndimensions:
      raise ValueError("Input array expected to have shape (N, {}).".format(self.sut.ndimensions))

    return np.ones(shape=(test.shape[0], 1))

