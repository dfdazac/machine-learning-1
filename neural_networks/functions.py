import numpy as np

def relu(X):
    """ ReLU activation function """
    return np.where(X > 0, X, 0)

def d_relu(X):
    """ Derivative of the ReLU activation function """
    return np.where(X > 0, 1, 0)

def one_hot_convert(t, n_classes):
    """ Convert arrays of integer labels to one-hot encoded vectors.
    Args:
        - t (n,): array containing n labels
        - n_classes (int): the number of possible classes
    Returns: (n_classes, n) binary array containing one-hot
        vectors in its columns.
    """
    if np.min(t) < 0 or np.max(t) >= n_classes:
        raise ValueError("Elements in array must be in the interval [0, {:d})".format(n_classes))
    T = np.zeros((n_classes, len(t)), dtype=int)
    T[t, np.arange(len(t))] = 1
    return T
