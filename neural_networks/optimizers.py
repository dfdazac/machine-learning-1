class SGD:
    """ Stochastic Gradient Descent optimizer for a model with Linear layers.
    Args:
        - model (NNClassifier): model with parameters to optimize
        - lr (float): learning rate
    """
    def __init__(self, model, lr):
        self.lr = lr
        self.model = model

    def step(self):
        for layer in self.model.layers:
            layer.W = layer.W - self.lr * layer.dW
            layer.b = layer.b - self.lr * layer.db
