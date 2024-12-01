import numpy as np

def phi(x):
    return 1 / (1 + np.exp(-x))

def dphi(x):
    return np.exp(-x) / (1 + np.exp(-x))**2

def dphiofphi(phi_value):
    return phi_value * (1 - phi_value)

def softmax(x):
    if x.ndim == 1:
        exp_x = np.exp(x - np.max(x))
        return exp_x / np.sum(exp_x)
    else:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
