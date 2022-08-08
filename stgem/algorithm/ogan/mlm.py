#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

class GeneratorNetwork(nn.Module):
    """
    Define the neural network model for the GAN generator.
    """

    def __init__(self, noise_dim, output_shape, neurons):
        super(GeneratorNetwork, self).__init__()

        # The dimension of the input vector.
        self.input_shape = noise_dim
        # The dimension of the output vector.
        self.output_shape = output_shape
        # Number of neurons per layer.
        self.neurons = neurons

        # We use three fully connected layers with self.neurons many neurons.
        self.glayer1 = nn.Linear(self.input_shape, self.neurons)
        self.glayer2 = nn.Linear(self.neurons, self.neurons)
        self.glayer3 = nn.Linear(self.neurons, self.output_shape)
        # Use uniform Glorot initialization of weights as in Keras.
        torch.nn.init.xavier_uniform_(self.glayer1.weight)
        torch.nn.init.xavier_uniform_(self.glayer2.weight)
        torch.nn.init.xavier_uniform_(self.glayer3.weight)

    def forward(self, x):
        """:meta private:"""
        x = F.relu(self.glayer1(x))
        x = F.relu(self.glayer2(x))
        x = torch.tanh(self.glayer3(x)) # Squash the output values to [-1, 1].

        return x

class DiscriminatorNetwork(nn.Module):
    """
    Define the neural network model for the GAN discriminator.
    """

    def __init__(self, input_shape, neurons, discriminator_output_activation="linear"):
        super(DiscriminatorNetwork, self).__init__()

        # The dimension of the input vector.
        self.input_shape = input_shape
        # Number of neurons per layer.
        self.neurons = neurons

        # We use three fully connected layers with self.neurons many neurons.
        self.dlayer1 = nn.Linear(self.input_shape, self.neurons)
        self.dlayer2 = nn.Linear(self.neurons, self.neurons)
        self.dlayer3 = nn.Linear(self.neurons, 1)
        # Use uniform Glorot initialization of weights as in Keras.
        torch.nn.init.xavier_uniform_(self.dlayer1.weight)
        torch.nn.init.xavier_uniform_(self.dlayer2.weight)
        torch.nn.init.xavier_uniform_(self.dlayer3.weight)

        # Select the output activation function.
        a = discriminator_output_activation
        if a == "linear":
            self.output_activation = torch.nn.Identity()
        elif a == "sigmoid":
            self.output_activation = torch.sigmoid
        else:
            raise Exception("Unknown output activation function '{}'.".format(a))

    def forward(self, x):
        """:meta private:"""
        x = F.leaky_relu(self.dlayer1(x), negative_slope=0.1) # LeakyReLU recommended in the literature for GANs discriminators.
        x = F.leaky_relu(self.dlayer2(x), negative_slope=0.1)
        x = self.output_activation(self.dlayer3(x))

        return x

class DiscriminatorNetwork1dConv(nn.Module):
    """
    Defines a neural network module for the GAN discriminator which uses 1D
    convolution. Useful when the test can be viewed as a time series.
    """

    def __init__(self, input_shape, feature_maps, kernel_sizes, convolution_activation, dense_neurons):
        """
        Creates a convolutional network with the following structure. For each
        number in the list feature_maps, create a 1D convolutional layer with
        the specified number of feature maps followed by a maxpool layer. The
        kernel sizes of the convolutional layer and the maxpool layer are
        specified by the first tuple in kernel_sizes. We use the specified
        activation function after each convolution layer. After the
        convolutions and maxpools, we use a single dense layer of the specified
        size with sigmoid activation.

        We always pad K-1 zeros when K is the kernel size. For now, we use a
        stride of 1.
        """

        super().__init__()

        # The dimension of the input vector.
        self.input_shape = input_shape
        # Number of feature maps.
        self.feature_maps = feature_maps
        # Corresponding kernel sizes.
        self.kernel_sizes = kernel_sizes
        # Number of neurons on the bottom dense layer.
        self.dense_neurons = dense_neurons

        activations = {"leaky_relu": F.leaky_relu,
                       "linear": nn.Identity(),
                       "relu": F.relu,
                       "sigmoid": torch.sigmoid,
                       "tanh": torch.tanh}

        # Convolution activation function.
        if not convolution_activation in activations:
            raise Exception("Unknown activation function '{}'.".format(convolution_activation))
        self.convolution_activation = activations[convolution_activation]

        # Define the convolutional layers and maxpool layers. Compute
        # simultaneously the number of inputs for the final dense layer by
        # feeding an input vector through the network.
        self.conv_layers = []
        self.maxpool_layers = []
        x = torch.zeros(1, 1, self.input_shape)
        C = nn.Conv1d(in_channels=1,
                      out_channels=feature_maps[0],
                      kernel_size=kernel_sizes[0][0],
                      padding=kernel_sizes[0][0]-1
                     )
        M = nn.MaxPool1d(kernel_size=kernel_sizes[0][1],
                         padding=kernel_sizes[0][1]-1
                        )
        x = M(C(x))
        self.conv_layers.append(C)
        self.maxpool_layers.append(M)
        for n, K in enumerate(feature_maps[1:]):
            C = nn.Conv1d(in_channels=feature_maps[n],
                          out_channels=K,
                          kernel_size=kernel_sizes[i+1][0],
                          padding=kernel_sizes[i+1][0]-1
                         )
            M = nn.MaxPool1d(kernel_size=kernel_sizes[i+1][1],
                             padding=kernel_sizes[i+1][1]-1
                            )
            x = M(C(x))
            self.conv_layers.append(C)
            self.maxpool_layers.append(M)

        # Define the final dense layer.
        self.flatten = nn.Flatten()
        I = x.reshape(-1).size()[0]
        self.dense_layer = nn.Linear(I, self.dense_neurons)
        self.bottom = nn.Linear(self.dense_neurons, 1)

    def forward(self, x):
        """:meta private:"""
        # Reshape to 1 channel.
        x = x.view(x.size()[0], 1, x.size()[1])
        for n in range(len(self.conv_layers)):
            C = self.conv_layers[n].to(x.device)
            M = self.maxpool_layers[n].to(x.device)
            x = self.convolution_activation(C(x))
            x = M(x)

        x = self.flatten(x)
        x = self.dense_layer(x)
        x = torch.sigmoid(self.bottom(x))

        return x

