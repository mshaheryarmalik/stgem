#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib

import numpy as np
import torch

from stgem import algorithm
from stgem.algorithm import Model

class OGAN_Model(Model):
    """
    Implements the WOGAN model.
    """

    default_parameters = {
        "optimizer": "Adam",
        "discriminator_lr": 0.005,
        "discriminator_betas": [0.9, 0.999],
        "generator_lr": 0.001,
        "generator_betas": [0.9, 0.999],
        "noise_batch_size": 512,
        "generator_loss": "MSE",
        "discriminator_loss": "MSE",
        "generator_mlm": "GeneratorNetwork",
        "generator_mlm_parameters": {
            "noise_dim": 20,
            "neurons": 64
        },
        "discriminator_mlm": "DiscriminatorNetwork",
        "discriminator_mlm_parameters": {
            "neurons": 64,
            "discriminator_output_activation": "sigmoid"
        },
        "train_settings_init": {
            "epochs": 2,
            "discriminator_epochs": 20,
            "generator_batch_size": 32
        },
        "train_settings": {
            "epochs": 1,
            "discriminator_epochs": 30,
            "generator_batch_size": 32
        }
    }

    def setup(self, sut, device, logger):
        super().setup(sut, device, logger)

        # Infer input and output dimensions for ML models.
        self.parameters["generator_mlm_parameters"]["output_shape"] = self.sut.idim
        self.parameters["discriminator_mlm_parameters"]["input_shape"] = self.sut.idim

        self._initialize()

    def _initialize(self):
        # Load the specified generator and discriminator machine learning
        # models and initialize them.
        module = importlib.import_module("stgem.algorithm.ogan.mlm")
        generator_class = getattr(module, self.generator_mlm)
        discriminator_class = getattr(module, self.discriminator_mlm)

        self.modelG = generator_class(**self.generator_mlm_parameters).to(self.device)
        self.modelD = discriminator_class(**self.discriminator_mlm_parameters).to(self.device)

        # Load the specified optimizers.
        module = importlib.import_module("torch.optim")
        optimizer_class = getattr(module, self.optimizer)
        generator_parameters = {k[10:]:v for k, v in self.parameters.items() if k.startswith("generator")}
        self.optimizerG = optimizer_class(self.modelG.parameters(), **algorithm.filter_arguments(generator_parameters, optimizer_class))
        discriminator_parameters = {k[14:]:v for k, v in self.parameters.items() if k.startswith("discriminator")}
        self.optimizerD = optimizer_class(self.modelD.parameters(), **algorithm.filter_arguments(discriminator_parameters, optimizer_class))

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
            self.lossG = get_loss(self.generator_loss)
            self.lossD = get_loss(self.discriminator_loss)
        except:
            raise

        # Setup loss saving.
        self.losses_D = []
        self.losses_G = []
        self.perf.save_history("discriminator_loss", self.losses_D)
        self.perf.save_history("generator_loss", self.losses_G)

    def reset(self):
        self._initialize()

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

                                   generator_batch_size: How large batches of
                                   noise are used at a training step.

                                 The default for each missing key is 1. Keys
                                 not found above are ignored.
        """

        if len(dataY) < len(dataX):
            raise ValueError("There should be at least as many training outputs as there are inputs.")

        dataX = torch.from_numpy(dataX).float().to(self.device)
        dataY = torch.from_numpy(dataY).float().to(self.device)

        # Unpack values from the train_settings dictionary.
        discriminator_epochs = train_settings["discriminator_epochs"] if "discriminator_epochs" in train_settings else 1
        generator_batch_size = train_settings["generator_batch_size"] if "generator_batch_size" in train_settings else 32

        # Save the training modes for restoring later.
        training_D = self.modelD.training
        training_G = self.modelG.training

        # Train the discriminator.
        # ---------------------------------------------------------------------
        # We want the discriminator to learn the mapping from tests to test
        # outputs.
        self.modelD.train(True)
        losses = []
        for _ in range(discriminator_epochs):
            D_loss = self.lossD(self.modelD(dataX), dataY)
            losses.append(D_loss.cpu().detach().numpy().item())
            self.optimizerD.zero_grad()
            D_loss.backward()
            self.optimizerD.step()

        self.losses_D += losses
        m = np.mean(losses)
        self.log("Discriminator epochs {}, Loss: {} -> {} (mean {})".format(discriminator_epochs, losses[0], losses[-1], m))

        self.modelD.train(False)

        # Visualize the computational graph.
        # print(make_dot(D_loss, params=dict(self.modelD.named_parameters())))

        # Train the generator on the discriminator.
        # -----------------------------------------------------------------------
        # We generate noise and label it to have output 0 (min objective).
        # Training the generator in this way should shift it to generate tests
        # with low output values (low objective). Notice that we need to
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

        fake_label = torch.zeros(size=(generator_batch_size, 1)).to(self.device)

        # Notice the following subtlety. Below the tensor 'outputs' contains
        # information on how it is computed (the computation graph is being kept
        # track off) up to the original input 'inputs' which does not anymore
        # depend on previous operations. Since 'self.modelD' is used as part of
        # the computation, its parameters are present in the computation graph.
        # These parameters are however not updated because the optimizer is
        # initialized only for the parameters of 'self.modelG' (see the
        # initialization of 'self.modelG'.

        losses = []
        for n in range(0, self.noise_batch_size, generator_batch_size):
            outputs = self.modelD(self.modelG(inputs[n:n+generator_batch_size]))
            G_loss = self.lossG(outputs, fake_label[:outputs.shape[0]])
            losses.append(G_loss.cpu().detach().numpy().item())
            self.optimizerG.zero_grad()
            G_loss.backward()
            self.optimizerG.step()

        self.losses_G += losses
        m = np.mean(losses)
        self.log("Generator steps {}, Loss: {} -> {}, mean {}".format(self.noise_batch_size//generator_batch_size + 1, losses[0], losses[-1], m))

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

        test_tensor = torch.from_numpy(test).float().to(self.device)
        return self.modelD(test_tensor).cpu().detach().numpy()

