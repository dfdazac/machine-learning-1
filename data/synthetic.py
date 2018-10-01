import numpy as np

def sin_normal_noise(N_samples, N_true=100, std=0.3):
    """
    Returns synthetic data of a sinusoidal signal with added
    normal noise.
    Args:
        N_samples: int, number of points
        N_true: int, number of points from signal without noise
        std: float, standard deviation of the added normal noise
    Returns:
        x_true: numpy array of size (N, 1) containing x data of signal
            without noise.
        t_true: numpy array of size (N, 1) containing target data of signal
            without noise.
        x: numpy array of size (N, 1) containing x data of signal
        t: numpy array of size (N, 1) containing target data of signal
    """
    # True function
    x_true = np.linspace(0, 1, N_true)[:, np.newaxis]
    t_true = np.sin(2*np.pi*x_true)

    # Training data
    x = np.linspace(0, 1, N_samples)[:, np.newaxis]
    t = np.sin(2*np.pi*x) + np.random.normal(0, std, N_samples)[:, np.newaxis]

    return x_true, t_true, x, t

def gaussian_mixture(N_samples):
    """Sample data from a Gaussian mixture with two components of mean -2 and 2
    and unit variance.
    Args:
        N_samples: int, the number of samples
    Returns:
        X: array, shape (N_samples,), the sampled data
    """
    X = np.empty(N_samples)
    # Choose from components with equal probability
    idx = np.random.choice([True, False], N_samples)
    n_1 = np.sum(idx)
    n_2 = N_samples - n_1
    # Fill array with samples from both components
    X[idx] = np.random.normal(-2, size=n_1)
    X[np.logical_not(idx)] = np.random.normal(2, size=n_2)
    return X
