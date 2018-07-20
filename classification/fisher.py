import numpy as np

class FisherDiscriminant:
    """
    A two-class Fisher's linear discriminant that can be used for classification.
    The fitted model separates the two classes so that within-class variance is
    minimized and between-class variance is maximized.
    """
    def __init__(self):
        self.weights_fit = False
        self.densities_fit = False

    def fit(self, X, Y):
        """
        Trains the model using the specified features and
        target values.
        Args:
            X: numpy array of shape (samples, features) containing
                input features.
            Y: numpy array of shape (samples,) and dtype np.int, containing
                binary labels from the set {0, 1}.
        """
        # Separate instances from each class
        x1, x2 = X[Y == 0], X[Y == 1]
        # Calculate means and center the data
        m1 = np.mean(x1, axis=0)
        m2 = np.mean(x2, axis=0)
        x1_m = x1 - m1
        x2_m = x2 - m2
        # Calculate within-class covariance matrix
        s_w = (x1_m.T @ x1_m) + (x2_m.T @ x2_m)
        # Calculate optimal weights and normalize
        self.weights = np.linalg.inv(s_w) @ (m2 - m1)
        self.weights = self.weights/np.linalg.norm(self.weights)
        self.weights_fit = True

        # Find ML estimates of class-conditional densities (Gaussians)
        # in the projected space
        X_proj = self.transform(X)
        X_p1, X_p2 = X_proj[Y == 0], X_proj[Y == 1]
        mean1 = np.mean(X_p1)
        std1 = np.std(X_p1)
        mean2 = np.mean(X_p2)
        std2 = np.std(X_p2)

        # Find threshold that minimizes misclassification rate
        a = 1/(2*std1**2) - 1/(2*std2**2)
        b = mean2/(std2**2) - mean1/(std1**2)
        c = mean1**2 /(2*std1**2) - mean2**2 / (2*std2**2) - np.log(std2/std1)
        solutions = np.roots([a, b, c])
        # Choose the solution closest to the midpoint of the means
        midpoint = (mean2 + mean1)/2
        self.threshold = solutions[np.argmin(np.abs(solutions - midpoint))]

        self.mean1, self.std1 = mean1, std1
        self.mean2, self.std2 = mean2, std2
        self.densities_fit = True

    def predict(self, X):
        """
        Returns a prediction for the specified data.
        Args:
            X: numpy array of shape (samples, features) containing
                input features.
        Returns: numpy array of shape (samples,) containing
                predictions.
        """
        if not self.densities_fit or not self.weights_fit:
            raise ValueError("Attempted to use untrained model.")
        # Project data and classify
        X_proj = self.transform(X)
        Y = np.zeros(len(X_proj), dtype=np.int)
        Y[X_proj > self.threshold] = 1
        return Y

    def transform(self, X):
        """
        Projects the data to a one-dimensional using the optimal projection
        according to Fisher's criterion.
        Args:
            X: numpy array of shape (samples, features) containing
                input features.
        Returns: numpy array of shape (samples,) containing
            the projected data.
        """
        if not self.weights_fit:
            raise ValueError("Attempted to use untrained model.")
        return X @ self.weights
