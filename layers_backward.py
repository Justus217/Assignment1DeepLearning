"""THWS/MAI/Introduction to Deep Learning - Assignment 2 Part 2 - layers with backward pass

Magda Gregorová, April 2026

This file extends the classes you implemented in Part 1.

Instructions:
  1. Copy your Linear, ReLU, and Model classes from layers.py (Part 1) into this file.
  2. Add the backward methods to Linear, ReLU, and Model as described below.
  3. Do not modify the forward methods.

Use only PyTorch tensor operations — no numpy, no torch.nn.
"""

import torch
from functions import linear_ggrad, relu_ggrad


# TODO: Copy your Linear class from Part 1 here and add the backward method below.

class Linear:
    """Linear layer: applies affine transformation y = X @ theta_1.T + theta_0.

    Attributes:
        theta_1: torch.tensor of shape (out_features, in_features) - weight matrix
        theta_0: torch.tensor of shape (1, out_features) - bias vector
        ins:     torch.tensor of shape (N, in_features) - stored input (set in forward)
        outs:    torch.tensor of shape (N, out_features) - stored output (set in forward)
    """

    def __init__(self, theta_1, theta_0):
        """Initialise the layer with weight matrix and bias vector.

        Args:
            theta_1: torch.tensor of shape (out_features, in_features)
            theta_0: torch.tensor of shape (1, out_features)
        """
        self.theta_0 = theta_0
        self.theta_1 = theta_1
        self.ins = None
        self.outs = None

    def forward(self, ins):
        """Forward pass: compute and return the affine transformation.

        Store the input in self.ins and the output in self.outs before returning.

        Args:
            ins: torch.tensor of shape (N, in_features)

        Returns:
            torch.tensor of shape (N, out_features)
        """
        self.ins = ins
        #self.outs = linear_forward(self.ins, self.theta_1, self.theta_0)
        self.outs = self.ins @ self.theta_1.T + self.theta_0
        return self.outs
        

    def backward(self, gout):
        """Backward pass: compute and store global gradients.

        Stores gradients as attributes on the tensors:
            self.theta_1.g of shape (out_features, in_features)
            self.theta_0.g of shape (1, out_features)
            self.ins.g     of shape (N, in_features)

        Args:
            gout: torch.tensor of shape (N, out_features) - upstream gradient dL/dZ

        Returns:
            torch.tensor of shape (N, in_features) - gradient w.r.t. input
        """
        gradients = linear_ggrad(gout, self.ins, self.theta_1, self.theta_0)
        self.theta_1.g, self.theta_0.g, self.ins.g = gradients
        return self.ins.g


# TODO: Copy your ReLU class from Part 1 here and add the backward method below.

class ReLU:
    """ReLU non-linearity: applies a(z) = max(0, z) element-wise.

    Attributes:
        ins:  torch.tensor - stored pre-activation input z (set in forward)
        outs: torch.tensor - stored activation a (set in forward)
    """

    def forward(self, ins):
        """Forward pass: apply ReLU element-wise.

        Store the input in self.ins and the output in self.outs before returning.

        Args:
            ins: torch.tensor of any shape - pre-activations z

        Returns:
            torch.tensor of same shape - activations a = relu(z)
        """
        self.ins = ins
        self.outs = torch.clamp(self.ins, min=0)
        return self.outs

    def backward(self, gout):
        """Backward pass: compute and store global gradient.

        Stores gradient as attribute on the tensor:
            self.ins.g of same shape as self.ins

        Args:
            gout: torch.tensor of same shape as self.ins - upstream gradient dL/dA

        Returns:
            torch.tensor of same shape - gradient w.r.t. input
        """
        # TODO: assign result to self.ins.g, return self.ins.g
        self.ins.g = relu_ggrad(gout, self.ins)
        return self.ins.g


# TODO: Copy your Model class from Part 1 here and add the backward method below.

class Model:
    """Neural network model: a sequence of layers applied in order.

    Attributes:
        layers: list of layer objects (Linear, ReLU, ...) in forward order
    """

    def __init__(self, layers):
        """Initialise with a list of layers.

        Args:
            layers: list of layer instances in the order of the forward pass
        """
        self.layers = layers

    def forward(self, ins):
        """Forward pass through all layers in order.

        Args:
            ins: torch.tensor of shape (N, in_features) - network input

        Returns:
            torch.tensor of shape (N, out_features) - network output
        """
        input = ins
        for layer in self.layers:
            output = layer.forward(input)
            input = output
        return output


    def backward(self, gout):
        """Backward pass through all layers in reverse order.

        Args:
            gout: torch.tensor - upstream gradient from the loss

        Returns:
            torch.tensor - gradient w.r.t. network input
        """
        # TODO: implement
        previous_grad = gout
        for layer in reversed(self.layers):
          previous_grad = layer.backward(previous_grad)
        return previous_grad
