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
