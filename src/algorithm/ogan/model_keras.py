#!/usr/bin/python3
# -*- coding: utf-8 -*-

import importlib

import numpy as np

from algorithm.model import Model as AlgModel
from keras.models import Sequential, Model
from keras.layers import Dense, LeakyReLU, Input
from tensorflow.keras.optimizers import Adam

class OGANK_Model(AlgModel):
    """
    Implements the OGAN model in Keras
    """

    def __init__(self, sut, parameters, logger=None):
        super().__init__(sut, parameters, logger)


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


        self.noise_dimensions=self.model_parameters["noise_dimensions"]
        lf = "mean_squared_error"
        sizeD =  self.model_parameters["d_size"]
        sizeG =  self.model_parameters["g_size"]
        input_shape = (self.input_dimension,)
        a = "relu"


        self.modelG = Sequential()
        self.modelG.add(Dense(sizeG, input_dim=self.noise_dimensions))
        self.modelG.add(Dense(sizeG, activation=a))
        self.modelG.add(Dense(sizeG, activation=a))
        self.modelG.add(Dense(self.input_dimension, activation="tanh"))

        self.modelG.compile(
            loss=lf,
            optimizer=Adam(learning_rate=self.model_parameters["g_adam_lr"]),
        )

        self.modelD = Sequential()
        self.modelD.add(Dense(sizeD, input_shape=input_shape, activation=a))
        self.modelD.add(Dense(sizeD, activation=a))
        self.modelD.add(Dense(sizeD, activation=a))
        self.modelD.add(Dense(1, activation="relu"))

        self.modelD.compile(loss=lf, optimizer=Adam(learning_rate= self.model_parameters["d_adam_lr"]))

        # Unpack values from the train_settings dictionary.
        discriminator_epochs = train_settings["discriminator_epochs"] if "discriminator_epochs" in train_settings else 1

        dloss = self.modelD.fit(
            # fix me
            dataX, dataY, epochs=discriminator_epochs, verbose=0
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
            loss=lf,
            optimizer=Adam(learning_rate=self.model_parameters["g_adam_lr"]),
        )

        batchSize =self.model_parameters["noise_bs"]
        noise = np.random.normal(0, 1, size=[batchSize, self.noise_dimensions])
        yGen = np.zeros(batchSize)

        gloss = self.gan.fit(noise, yGen, epochs=train_settings["generator_epochs"], verbose=0)



    def generate_test(self, N=1):
        """
        Generate N random tests.

        Args:
          N (int): Number of tests to be generated.

        Returns:
          output (np.ndarray): Array of shape (N, self.sut.ndimensions).
        """

        noise = np.random.normal(0, 1, size=(N, self.noise_dimensions))
        tests = self.modelG.predict(noise)
        return tests



    def predict_objective(self, test):
        """
        Predicts the objective function value of the given tests.

        Args:
          test (np.ndarray): Array of shape (N, self.sut.ndimensions).

        Returns:
          output (np.ndarray): Array of shape (N, 1).
        """
        return self.modelD.predict(test)


