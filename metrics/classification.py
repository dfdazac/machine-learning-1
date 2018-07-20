import numpy as np

def accuracy(y_true, y_pred):
    """
	Returns the accuracy of the predicted values given the true values.
	Args:
		y_true: numpy array of size (samples,).
		y_pred: numpy array of size (samples,).
	Returns:
		float, the accuracy.
	"""
    if y_true.shape != y_pred.shape:
        raise ValueError('Arrays have not the same size')

    correct = np.sum(y_true == y_pred)
    return correct/len(y_true)
