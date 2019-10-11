"""This file contains the :class: `Model` to compile a Keras model with
the user-defined configuration

Author:
-------
Aashish Yadavally
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.constraints import maxnorm


class Model:
    """Keras Sequential model for the given configuration

    Parameters
    -----------
        input_dim (int):
            Number of dimensions of input
        dropout (float):
            Dropout regularization ratio
        kernel_constraint (int):
            Kernel constraint in the network layers
        activation (str):
            Activation function for the hidden layers
        optimizer (str):
            Optimization technique for compiling model
    """
    def __init__(self, input_dim,
                 dropout=0.3,
                 kernel_constraint=3,
                 activation='sigmoid',
                 optimizer='adam'):
        """Initializes :class: `Model`
        """
        self.input_dim = input_dim
        self.dropout = dropout
        self.kernel_constraint = kernel_constraint
        self.activation = activation
        self.optimizer = optimizer

    def define(self):
        """Defines a Keras Sequential model with the given configuration

        Returns
        --------
            model (keras.models.Sequential)
                Model compiled on the given configuration
        """
        model = Sequential()
        model.add(Dense(100, activation=self.activation,
                        input_dim=self.input_dim,
                        kernel_constraint=maxnorm(self.kernel_constraint)))
        model.add(Dropout(self.dropout))
        model.add(Dense(36, activation='softmax',
                        kernel_constraint=maxnorm(self.kernel_constraint)))
        model.compile(loss='categorical_crossentropy',
                      optimizer=self.optimizer,
                      metrics=['accuracy'])
        return model
