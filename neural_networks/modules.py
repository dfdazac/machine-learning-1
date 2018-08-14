import numpy as np
from .functions import relu, d_relu, one_hot_convert

class Linear:
    """ A linear layer that performs the operation
    W X + b
    Args:
        - n_inputs (int): number of inputs
        - n_outputs (int): number of outputs
    """
    def __init__(self, n_inputs, n_outputs):
        self.W = np.random.standard_normal((n_outputs, n_inputs)) * np.sqrt(2/n_inputs)
        self.b = np.zeros((n_outputs, 1))
        self.dW = None
        self.db = None
        self.A_prev = None
        self.Zl = None

    def __call__(self, X, cache=False):
        """ Performs a forward pass on the layer given X.
        Args:
            - X (n_inputs, n): input array
            - cache (bool): whether or not to save intermediate
                values. Used during backpropagation.
        Returns: the result of the operation of the layer on X.
        """
        Zl = self.W @ X + self.b
        if cache:
            self.A_prev = X.copy()
            self.Zl = Zl.copy()
        return Zl

class NNClassifier:
    """ A neural network classifier.
    Args:
        - units (iterable): contains the number of units (int)
            for each layer, from the input layer up to the
            output layer.
    """
    def __init__(self, units):
        self.units = units
        self.n_classes = units[-1]
        self.layers = []
        for i in range(1, len(units)):
            self.layers.append(Linear(units[i - 1], units[i]))
        self.dZL = None

    def forward(self, X, cache=False):
        A_prev = X
        for layer in self.layers:
            Zl = layer(A_prev, cache)
            A_prev = relu(Zl)

        return Zl

    def _log_normalizer(self, ZL):
        max_ZL = np.max(ZL, axis=0, keepdims=True)
        log_Z = max_ZL + np.log(np.sum(np.exp(ZL - max_ZL), axis=0, keepdims=True))
        return log_Z

    def loss(self, X, t):
        ZL = self.forward(X, cache=True)
        log_Z = self._log_normalizer(ZL)
        log_probs = ZL - log_Z

        T = one_hot_convert(t, self.n_classes)
        Z = np.exp(log_Z)
        self.dZL = np.exp(ZL)/Z - T

        return -np.mean(log_probs[T == 1])

    def forward_probs(self, X):
        ZL = self.forward(X)
        log_Z = self._log_normalizer(ZL)
        log_probs = ZL - log_Z
        return np.exp(log_probs)

    def backward(self):
        # Number of samples in mini-batch
        n = self.dZL.shape[1]
        for i, layer in enumerate(reversed(self.layers)):
            if i == 0:
                dZl = self.dZL
            else:
                dZl = d_relu(layer.Zl) * dAl
            layer.dW = 1/n * (dZl @ layer.A_prev.T)
            layer.db = np.mean(dZl, axis=1, keepdims=True)
            dAl = layer.W.T @ dZl
