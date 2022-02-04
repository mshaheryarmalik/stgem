#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib

import numpy as np
import torch

import algorithm
from algorithm.model import Model

class OGAN_Model(Model):
    """
    Implements the WOGAN model.
    """

    def __init__(self, sut, parameters, logger=None):
        super().__init__(sut, parameters, logger)

        self.noise_batch_size = self.ogan_model_parameters["noise_batch_size"]

        # Load the specified generator and discriminator machine learning
        # models and initialize them.
        module = importlib.import_module(".mlm", "algorithm.ogan")
        generator_class = getattr(module, self.generator_mlm)
        discriminator_class = getattr(module, self.discriminator_mlm)
        self.modelG = generator_class(**self.generator_mlm_parameters).to(self.device)
        self.modelD = discriminator_class(**self.discriminator_mlm_parameters).to(self.device)

        # Load the specified optimizers.
        module = importlib.import_module("torch.optim")
        optimizer_class = getattr(module, self.ogan_model_parameters["optimizer"])
        self.optimizerG = optimizer_class(self.modelG.parameters(), **algorithm.filter_arguments(self.ogan_model_parameters, optimizer_class))
        self.optimizerD = optimizer_class(self.modelD.parameters(), **algorithm.filter_arguments(self.ogan_model_parameters, optimizer_class))

        # Loss functions.
        def get_loss(loss_s):
            loss_s = loss_s.lower()
            if loss_s == "mse":
                loss = torch.nn.MSELoss()
            elif loss_s == "l1":
                loss = torch.nn.L1Loss()
            elif loss_s == "mse,logit" or loss_s == "l1,logit":
                # When doing regression with values in [0, 1], we can use a
                # logit transformation to map the values from [0, 1] to \R
                # to make errors near 0 and 1 more drastic. Since logit is
                # undefined in 0 and 1, we actually first transform the values
                # to the interval [0.01, 0.99].
                if loss_s == "mse,logit":
                    g = torch.nn.MSELoss()
                else:
                    g = torch.nn.L1Loss()
                def f(X, Y):
                    return g(torch.logit(0.98*X + 0.01), torch.logit(0.98*Y + 0.01))
                loss = f
            else:
                raise Exception("Unknown loss function '{}'.".format(loss_s))

            return loss

        try:
            self.lossG = get_loss(self.ogan_model_parameters["generator_loss"])
            self.lossD = get_loss(self.ogan_model_parameters["discriminator_loss"])
        except:
            raise

    def train_with_batch(self, dataX, dataY, train_settings):
        """
        Train the OGAN with a batch of training data.

        Args:
          dataX (np.ndarray): Array of tests of shape
                              (N, self.input_dimension).
          dataY (np.ndarray): Array of test outputs of shape (N, 1).
          train_settings (dict): A dictionary setting up the number of training
                                 epochs for various parts of the model. The
                                 keys are as follows:

                                   discriminator_epochs: How many times the
                                   discriminator is trained per call.

                                   generator_epochs: How many times the
                                   generator is trained per call.

                                 The default for each missing key is 1. Keys
                                 not found above are ignored.
        """

        if len(dataY) < len(dataX):
            raise ValueError("There should be at least as many training outputs as there are inputs.")

        dataX = torch.from_numpy(dataX).float().to(self.device)
        dataY = torch.from_numpy(dataY).float().to(self.device)

        # Unpack values from the epochs dictionary.
        discriminator_epochs = train_settings["discriminator_epochs"] if "discriminator_epochs" in train_settings else 1
        generator_epochs = train_settings["generator_epochs"] if "generator_epochs" in train_settings else 1

        # Save the training modes for restoring later.
        training_D = self.modelD.training
        training_G = self.modelG.training

        # Train the discriminator.
        # ---------------------------------------------------------------------
        # We want the discriminator to learn the mapping from tests to test
        # outputs.
        self.modelD.train(True)
        for n in range(discriminator_epochs):
            D_loss = self.lossD(self.modelD(dataX), dataY)
            self.optimizerD.zero_grad()
            D_loss.backward()
            self.optimizerD.step()

            self.log("Discriminator epoch {}/{}, Loss: {}".format(n + 1, discriminator_epochs, D_loss))

        self.modelD.train(False)

        # Visualize the computational graph.
        # print(make_dot(D_loss, params=dict(self.modelD.named_parameters())))

        # Train the generator on the discriminator.
        # -----------------------------------------------------------------------
        # We generate noise and label it to have output 1 (max objective).
        # Training the generator in this way should shift it to generate tests
        # with high output values (high objective). Notice that we need to
        # validate the generated tests as no invalid tests with high fitness
        # exist.
        self.modelG.train(False)
        inputs = np.zeros(shape=(self.noise_batch_size, self.modelG.input_shape))
        k = 0
        while k < inputs.shape[0]:
            noise = torch.rand(1, self.modelG.input_shape)*2 - 1
            new_test = self.modelG(noise.to(self.device)).cpu().detach().numpy()
            if self.sut.validity(new_test) == 0: continue
            inputs[k,:] = noise[0,:]
            k += 1
        self.modelG.train(True)
        inputs = torch.from_numpy(inputs).float().to(self.device)

        fake_label = torch.ones(size=(self.noise_batch_size, 1)).to(self.device)

        # Notice the following subtlety. Below the tensor 'outputs' contains
        # information on how it is computed (the computation graph is being kept
        # track off) up to the original input 'inputs' which does not anymore
        # depend on previous operations. Since 'self.modelD' is used as part of
        # the computation, its parameters are present in the computation graph.
        # These parameters are however not updated because the optimizer is
        # initialized only for the parameters of 'self.modelG' (see the
        # initialization of 'self.modelG'.

        for n in range(generator_epochs):
            outputs = self.modelD(self.modelG(inputs))
            G_loss = self.lossG(outputs, fake_label)
            self.optimizerG.zero_grad()
            G_loss.backward()
            self.optimizerG.step()
            self.log("Generator epoch: {}/{}, Loss: {}".format(n + 1, generator_epochs, G_loss))

        self.modelG.train(False)

        # Visualize the computational graph.
        # print(make_dot(G_loss, params=dict(self.modelG.named_parameters())))

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

        training_G = self.modelG.training
        # Generate uniform noise in [-1, 1].
        noise = (torch.rand(size=(N, self.modelG.input_shape))*2 - 1).to(self.device)
        self.modelG.train(False)
        result = self.modelG(noise).cpu().detach().numpy()
        self.modelG.train(training_G)
        return result

    def predict_objective(self, test):
        """
        Predicts the objective function value of the given tests.

        Args:
          test (np.ndarray): Array of shape (N, self.sut.ndimensions).

        Returns:
          output (np.ndarray): Array of shape (N, 1).
        """

        if len(test.shape) != 2 or test.shape[1] != self.sut.ndimensions:
            raise ValueError("Input array expected to have shape (N, {}).".format(self.sut.ndimensions))

        test_tensor = torch.from_numpy(test).float().to(self.device)
        return self.modelD(test_tensor).cpu().detach().numpy()

