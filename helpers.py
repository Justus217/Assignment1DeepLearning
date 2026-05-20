"""THWS/MAI/Introduction to Deep Learning - Assignment 2 - helper functions

Magda Gregorová, April 2026

These functions are provided. You do not need to modify this file.
"""

import csv
import torch


def load_data(filename='./ann_data/data.csv'):
    """Load and return the dataset as tensors.

    Args:
        filename: path to the CSV file

    Returns:
        X: torch.tensor of shape (N, 4) - input features (standardised)
        y: torch.tensor of shape (N, 1) - target values (standardised)
    """
    with open(filename, newline='') as f:
        reader = csv.DictReader(f)
        rows = [row for row in reader]

    X = torch.tensor([[float(r['age']), float(r['size']),
                       float(r['distance']), float(r['floor'])] for r in rows])
    y = torch.tensor([[float(r['price'])] for r in rows])
    return X, y


def numerical_gradient(f, x, h=1e-4):
    """Compute numerical gradient of scalar-valued f at x using central differences.

    Args:
        f: function that takes a tensor of same shape as x and returns a scalar
        x: torch.tensor - point at which to evaluate the gradient
        h: float - step size for finite differences

    Returns:
        grad: torch.tensor of same shape as x - numerical gradient
    """
    x_flat = x.flatten()
    grad_flat = torch.zeros_like(x_flat)
    for i in range(x_flat.numel()):
        x_plus, x_minus = x_flat.clone(), x_flat.clone()
        x_plus[i]  += h
        x_minus[i] -= h
        grad_flat[i] = (f(x_plus.view_as(x)) - f(x_minus.view_as(x))) / (2 * h)
    return grad_flat.view_as(x)


def grad_checker(grad_analytic, grad_numeric, name=''):
    """Compare analytic and numerical gradients and print a summary.

    Args:
        grad_analytic: torch.tensor - analytically computed gradient
        grad_numeric:  torch.tensor - numerically computed gradient
        name:          str - label for the printout
    """
    abs_err = (grad_analytic - grad_numeric).abs()
    rel_err = abs_err / (grad_numeric.abs() + 1e-8)
    label = f'[{name}] ' if name else ''
    print(f'{label}max absolute error: {abs_err.max():.2e}')
    print(f'{label}max relative error: {rel_err.max():.2e}')
    if rel_err.max() < 1e-3:
        print(f'{label}Gradient check PASSED.')
    else:
        print(f'{label}Gradient check FAILED — check your implementation.')
