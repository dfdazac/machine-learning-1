import numpy as np

class LinearRegression:
    """
    A linear regression model with polynomial basis functions.
    Args:
        lamb: float, regularization factor.
    """
    def __init__(self, lamb=0):
        self.lamb = lamb
        self.weights = None

    def fit(self, X, Y):
        """
        Trains the model using the specified features and
        target values.
        Args:
            X: numpy array of shape (samples, features) containing
                input features.
            Y: numpy array of shape (samples, targets) containing
                target values.
        """
        n_samples, n_features = X.shape
        n_targets = Y.shape[1]

        # Pseudo-inverse matrix
        p_inv = X.T @ X
        # Add regularizing term if required
        if self.lamb > 0:
            p_inv += self.lamb * np.eye(n_features)
        p_inv = np.linalg.inv(p_inv)

        # Solve the normal equation
        self.weights = p_inv @ X.T @ Y

    def predict(self, X):
        """
        Returns a prediction for the specified data.
        Args:
            X: numpy array of shape (samples, features) containing
                input features.
        Returns: numpy array of shape (samples, targets) containing
                predictions.
        """
        if self.weights is None:
            raise ValueError("Attempted to predict before training model.")
        return X @ self.weights
