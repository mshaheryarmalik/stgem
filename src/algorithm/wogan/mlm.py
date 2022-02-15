#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class WOGAN_NN(nn.Module):
    """
    Base class for simple dense neural networks.
    """

    def __init__(self, input_shape, hidden_neurons, output_shape, output_activation, hidden_activation, batch_normalization=False, layer_normalization=False):
        super().__init__()

        # The dimension of the input vector.
        self.input_shape = input_shape
        # The dimension of the output vector.
        self.output_shape = output_shape
        # List of numbers of neurons in the hidden layers.
        self.hidden_neurons = hidden_neurons
        # Use batch normalization before each activation (except the last one).
        self.batch_normalization = batch_normalization
        # Use layer normalization before each activation (except the last one).
        self.layer_normalization = layer_normalization

        if self.batch_normalization and self.layer_normalization:
            raise Exception("Cannot use both batch normalization and layer normalization (not recommended).")

        # Map for activations.
        activations = {"leaky_relu": F.leaky_relu,
                       "linear": nn.Identity(),
                       "relu": F.relu,
                       "sigmoid": torch.sigmoid,
                       "tanh": torch.tanh}

        # Hidden layer activation.
        if not hidden_activation in activations:
            raise Exception("Unknown activation function '{}'.".format(activation))
        self.hidden_activation = activations[hidden_activation]

        # Output activation.
        if not output_activation in activations:
            raise Exception("Unknown activation function '{}'.".format(output_activation))
        self.output_activation = activations[output_activation]

        # We use fully connected layers with the specified number of neurons.
        self.top = nn.Linear(self.input_shape, self.hidden_neurons[0])
        self.hidden = []
        if self.batch_normalization:
            self.norm = [nn.BatchNorm1d(self.hidden_neurons[0])]
        if self.layer_normalization:
            self.norm = [nn.LayerNorm(self.hidden_neurons[0])]
        for i, neurons in enumerate(self.hidden_neurons[1:]):
            self.hidden.append(nn.Linear(self.hidden_neurons[i], neurons))
            if self.batch_normalization:
                self.norm.append(nn.BatchNorm1d(neurons))
            if self.layer_normalization:
                self.norm.append(nn.LayerNorm(neurons))
        self.bottom = nn.Linear(self.hidden_neurons[-1], self.output_shape)

    def forward(self, x):
        x = self.hidden_activation(self.top(x))
        for i, L in enumerate(self.hidden):
            L = L.to(x.device)
            if self.batch_normalization or self.layer_normalization:
                x = self.hidden_activation(self.norm[i](L(x)))
            else:
                x = self.hidden_activation(L(x))
        x = self.output_activation(self.bottom(x))

        return x

class AnalyzerNetwork(WOGAN_NN):
    """
    Define a regression neural network model for the WOGAN analyzer.
    """

    def __init__(self, input_shape, hidden_neurons, hidden_activation="relu", layer_normalization=False):
        super().__init__(input_shape=input_shape,
                         hidden_neurons=hidden_neurons,
                         output_shape=1,
                         output_activation="sigmoid",
                         hidden_activation=hidden_activation,
                         layer_normalization=layer_normalization
                        )
    
class AnalyzerNetwork_classifier(WOGAN_NN):
    """
    Define a classification neural network model for the WOGAN analyzer.
    """

    def __init__(self, classes, input_shape, hidden_neurons):
        # Number of classes.
        self.classes = classes
        super().__init__(input_shape=input_shape,
                         hidden_neurons=hidden_neurons,
                         output_shape=self.classes,
                         output_activation="linear",
                         hidden_activation="relu",
                         batch_normalization=True
                        )

class CriticNetwork(WOGAN_NN):
    """
    Define the neural network model for the WGAN critic.
    """

    def __init__(self, input_shape, hidden_neurons):
        super().__init__(input_shape=input_shape,
                         hidden_neurons=hidden_neurons,
                         output_shape=1,
                         output_activation="linear",
                         hidden_activation="leaky_relu"
                        )

class GeneratorNetwork(WOGAN_NN):
    """
    Define the neural network model for the WGAN generator.
    """

    def __init__(self, noise_dim, hidden_neurons, output_shape, batch_normalization=False, layer_normalization=False):
        super().__init__(input_shape=noise_dim,
                         hidden_neurons=hidden_neurons,
                         output_shape=output_shape,
                         output_activation="tanh",
                         hidden_activation="relu",
                         batch_normalization=batch_normalization,
                         layer_normalization=layer_normalization
                        )

