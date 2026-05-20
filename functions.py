"""THWS/MAI/Introduction to Deep Learning - Assignment 2 Part 2 - functions

Magda Gregorová, April 2026

Implement the functions marked with TODO.
Do not change function signatures.
Use only PyTorch tensor operations — no numpy, no torch.nn.

For MSE, the local and global gradients are always identical since MSE is the
root of the computation graph — there is no upstream gradient to multiply by.
We therefore provide a single backward function for MSE.

For Linear and ReLU, the backward pass is split into two functions:
  _lgrad: computes local gradients (partial derivatives of this node's output
          w.r.t. its inputs), independent of the rest of the graph
  _ggrad: computes global gradients by applying the chain rule with upstream
          gout, and returns the results
"""

import torch


# ==============================================================================
# Section 1 — MSE backward
# ==============================================================================

def mse_backward_scalar(y_pred, y):
    """Gradient of scalar MSE loss w.r.t. prediction.

    Since MSE is the root of the computation graph, local and global gradients
    are always identical.

    Args:
        y_pred: float - predicted value
        y:      float - true value

    Returns:
        float - gradient dL/d(y_pred)
    """
    # Grad: 2 * (y_pred - y) * 1
    
    return 2 * (y_pred - y)


def mse_backward(y_pred, y):
    """Gradient of batch MSE loss w.r.t. each prediction.

    Since MSE is the root of the computation graph, local and global gradients
    are always identical.

    Args:
        y_pred: torch.tensor of shape (N, 1)
        y:      torch.tensor of shape (N, 1)

    Returns:
        torch.tensor of shape (N, 1)
    """
    # Grad: 2/N * (y_pred_i - y_i)
    N = len(y)
    return 2/N * (y_pred - y)
    


# ==============================================================================
# Section 2 — Linear backward
# ==============================================================================

def linear_lgrad_scalar(x, theta):
    """Local gradients for scalar linear function z = theta_1 * x + theta_0.

    Args:
        x:     torch.tensor of shape () - scalar input
        theta: torch.tensor of shape (2,) - (theta_0, theta_1)

    Returns:
        tuple: (lgrad_theta_0, lgrad_theta_1, lgrad_x)
               lgrad_theta_0: torch.tensor of shape ()
               lgrad_theta_1: torch.tensor of shape ()
               lgrad_x:       torch.tensor of shape ()

    Hint: use .clone() when returning parts of existing tensors.
    """
    lgrad_theta_0 = torch.tensor(1)
    lgrad_theta_1 = x.clone()
    lgrad_x = theta.clone()[1]
    
    return (lgrad_theta_0, lgrad_theta_1, lgrad_x)


def linear_ggrad_scalar(gout, x, theta):
    """Global gradients for scalar linear function z = theta_1 * x + theta_0.

    Applies the chain rule with upstream gout.
    In the scalar case this is elementwise: gout * lgrad.

    Args:
        gout:  torch.tensor of shape () - upstream gradient dL/dz
        x:     torch.tensor of shape () - scalar input
        theta: torch.tensor of shape (2,) - (theta_0, theta_1)

    Returns:
        tuple: (ggrad_theta_0, ggrad_theta_1, ggrad_x)
               each a torch.tensor of shape ()
    """
    lgrad_theta_0, lgrad_theta_1, lgrad_x = linear_lgrad_scalar(x, theta)
    ggrad_theta_0 = gout * lgrad_theta_0
    ggrad_theta_1 = gout * lgrad_theta_1
    ggrad_x = gout * lgrad_x
    return (ggrad_theta_0, ggrad_theta_1, ggrad_x)


def linear_lgrad(ins, theta_1, theta_0):
    """Local gradient factors for batched linear layer Z = X @ theta_1.T + theta_0.

    Args:
        ins:     torch.tensor of shape (N, in_features)
        theta_1: torch.tensor of shape (out_features, in_features)
        theta_0: torch.tensor of shape (1, out_features)

    Returns:
        tuple: (lgrad_theta_1_factor, lgrad_theta_0_factor, lgrad_ins)
               lgrad_theta_1_factor: torch.tensor of shape (N, in_features)
               lgrad_theta_0_factor: torch.tensor of shape (N, out_features)
               lgrad_ins:            torch.tensor of shape (out_features, in_features)
    """
    lgrad_theta_1_factor = ins.clone()
    lgrad_theta_0_factor = torch.ones(ins.shape[0], theta_0.shape[1])
    lgrad_ins = theta_1.clone()
    return (lgrad_theta_1_factor, lgrad_theta_0_factor, lgrad_ins)


def linear_ggrad(gout, ins, theta_1, theta_0):
    """Global gradients for batched linear layer Z = X @ theta_1.T + theta_0.

    Applies the chain rule with upstream gout.
    In the matrix case this involves matrix products — think carefully about
    how the dimensions work out.

    Args:
        gout:    torch.tensor of shape (N, out_features) - upstream gradient dL/dZ
        ins:     torch.tensor of shape (N, in_features)
        theta_1: torch.tensor of shape (out_features, in_features)
        theta_0: torch.tensor of shape (1, out_features)

    Returns:
        tuple: (ggrad_theta_1, ggrad_theta_0, ggrad_ins)
               ggrad_theta_1: torch.tensor of shape (out_features, in_features)
               ggrad_theta_0: torch.tensor of shape (1, out_features)
               ggrad_ins:     torch.tensor of shape (N, in_features)
    """
    lgrad_t1, lgrad_t0, lgrad_ins = linear_lgrad(ins, theta_1, theta_0)
    
    ggrad_theta_1 = gout.T @ lgrad_t1
    ggrad_theta_0 =  gout.sum(dim=0, keepdim=True)
    ggrad_ins = gout @ lgrad_ins
    
    return(ggrad_theta_1, ggrad_theta_0, ggrad_ins)


# ==============================================================================
# Section 3 — ReLU backward
# ==============================================================================

def relu_lgrad_scalar(z):
    """Local gradient for scalar ReLU: a = relu(z) = max(0, z).

    Args:
        z: torch.tensor of shape () - scalar pre-activation

    Returns:
        torch.tensor of shape () - local gradient da/dz
    """
    if z > 0:
        return torch.tensor(1)
    else: return torch.tensor(0)


def relu_ggrad_scalar(gout, z):
    """Global gradient for scalar ReLU.

    Applies the chain rule with upstream gout.

    Args:
        gout: torch.tensor of shape () - upstream gradient dL/da
        z:    torch.tensor of shape () - scalar pre-activation

    Returns:
        torch.tensor of shape () - global gradient dL/dz
    """
    return gout * relu_lgrad_scalar(z)


def relu_lgrad(ins):
    """Local gradient for batch ReLU: A = relu(Z), element-wise.

    Args:
        ins: torch.tensor of any shape - pre-activations Z

    Returns:
        torch.tensor of same shape - local gradient
    """
    return (ins > 0).float()


def relu_ggrad(gout, ins):
    """Global gradient for batch ReLU.

    Applies the chain rule with upstream gout.

    Args:
        gout: torch.tensor of same shape as ins - upstream gradient dL/dA
        ins:  torch.tensor of any shape - pre-activations Z

    Returns:
        torch.tensor of same shape - global gradient dL/dZ
    """
    
    return gout * relu_lgrad(ins)