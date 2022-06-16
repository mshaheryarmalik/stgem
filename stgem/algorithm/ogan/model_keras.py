#!/usr/bin/python3
# -*- coding: utf-8 -*-

import dill as pickle
import numpy as np

from stgem.algorithm import Model as AlgModel
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

class OGANK_Model(AlgModel):
    """
    Implements the OGAN model in Keras
    """

    default_parameters = {
        "optimizer": "Adam",
        "d_size": 512,
        "g_size": 512,
        "d_adam_lr": 0.001,
        "g_adam_lr": 0.0001,
        "lossfunction": "mean_squared_error",
        "noise_dimensions": 50,
        "noise_batch_size": 10000,
        "train_settings_init": {"epochs": 1, "discriminator_epochs": 10, "generator_epochs": 1},
        "train_settings": {"epochs": 1, "discriminator_epochs": 10, "generator_epochs": 1}
    }

    def init_model(self):
        sizeD = self.d_size
        sizeG = self.g_size
        input_shape = (self.input_dimension, )
        a = "relu"

        self.modelG = Sequential()
        self.modelG.add(Dense(sizeG, input_dim=self.noise_dimensions))
        self.modelG.add(Dense(sizeG, activation=a))
        self.modelG.add(Dense(sizeG, activation=a))
        self.modelG.add(Dense(self.search_space.input_dimension, activation="tanh"))

        self.modelG.compile(
            loss=self.lossfunction,
            optimizer=Adam(learning_rate=self.g_adam_lr),
        )

        self.modelD = Sequential()
        self.modelD.add(Dense(sizeD, input_shape=input_shape, activation=a))
        self.modelD.add(Dense(sizeD, activation=a))
        self.modelD.add(Dense(sizeD, activation=a))
        self.modelD.add(Dense(1, activation="relu"))

        self.modelD.compile(loss=self.lossfunction, optimizer=Adam(learning_rate=self.d_adam_lr))

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

        self.init_model()

        # Unpack values from the train_settings dictionary.
        discriminator_epochs = train_settings["discriminator_epochs"] if "discriminator_epochs" in train_settings else 1

        dloss = self.modelD.fit(
            # fix me
            dataX, dataY, epochs=discriminator_epochs, verbose=False
        )

        for l in self.modelD.layers:
            l.trainable = False
        self.modelD.trainable = False

        ganInput = Input(shape=( self.noise_dimensions,))
        self.gan = Model(
            inputs=ganInput,
            outputs=self.modelD(self.modelG(ganInput)),
        )
        self.gan.compile(
            loss=self.lossfunction,
            optimizer=Adam(learning_rate=self.g_adam_lr),
        )

        batchSize =self.noise_batch_size
        noise = np.random.normal(0, 1, size=[batchSize, self.noise_dimensions])
        yGen = np.zeros(batchSize)

        gloss = self.gan.fit(noise, yGen, epochs=train_settings["generator_epochs"], verbose=True)

    def generate_test(self, N=1):
        """
        Generate N random tests.

        Args:
          N (int): Number of tests to be generated.

        Returns:
          output (np.ndarray): Array of shape (N, self.n_inputs).
        """

        noise = np.random.normal(0, 1, size=(N, self.noise_dimensions))
        tests = self.modelG.predict(noise)
        return tests

    def predict_objective(self, test):
        """
        Predicts the objective function value of the given tests.

        Args:
          test (np.ndarray): Array of shape (N, self.n_inputs).

        Returns:
          output (np.ndarray): Array of shape (N, 1).
        """
        return self.modelD.predict(test)

