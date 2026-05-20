"""THWS/MAI/Introduction to Deep Learning - Assignment 2 - plotting

Magda Gregorová, April 2026

These functions are provided. You do not need to modify this file.
"""

import torch
import matplotlib.pyplot as plt


def plot_data(X, y):
    """Plot size vs price and distance vs price side by side.

    Args:
        X: torch.tensor of shape (N, 4) - input features (age, size, distance, floor)
        y: torch.tensor of shape (N, 1) - target prices
    """
    x_size     = X[:, 1].numpy()
    x_distance = X[:, 2].numpy()
    y_vals     = y[:, 0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].scatter(x_size, y_vals, alpha=0.4, s=15, color='steelblue')
    axes[0].set_xlabel('size (standardised)')
    axes[0].set_ylabel('price (standardised)')
    axes[0].set_title('Size vs Price')

    axes[1].scatter(x_distance, y_vals, alpha=0.4, s=15, color='steelblue')
    axes[1].set_xlabel('distance (standardised)')
    axes[1].set_ylabel('price (standardised)')
    axes[1].set_title('Distance vs Price')

    fig.tight_layout()
    plt.show()


def plot_fit(X, y, model):
    """Plot model fit against each input feature (2x2 grid).

    For each feature, varies that feature across its observed range while
    holding all others fixed at zero (their mean, since data is standardised).

    Args:
        X:     torch.tensor of shape (N, 4) - input features (age, size, distance, floor)
        y:     torch.tensor of shape (N, 1) - target prices
        model: object with a forward(X) method returning shape (N, 1)
    """
    feature_names = ['age', 'size', 'distance', 'floor']
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes = axes.flatten()

    for j, (ax, name) in enumerate(zip(axes, feature_names)):
        # scatter: actual data points
        order = X[:, j].argsort()
        x_feat = X[order, j].numpy()
        y_vals = y[order, 0].numpy()
        ax.scatter(x_feat, y_vals, alpha=0.4, s=15, color='steelblue', label='data')

        # prediction line: vary feature j, hold others at 0
        grid = torch.linspace(X[:, j].min(), X[:, j].max(), 200)
        X_grid = torch.zeros(200, 4)
        X_grid[:, j] = grid
        with torch.no_grad():
            y_grid = model.forward(X_grid)[:, 0].numpy()
        ax.plot(grid.numpy(), y_grid, color='crimson', linewidth=2, label='model fit')

        ax.set_xlabel(f'{name} (standardised)')
        ax.set_ylabel('price (standardised)')
        ax.set_title(f'{name.capitalize()} vs price')
        ax.legend(fontsize=8)

    fig.tight_layout()
    plt.show()
