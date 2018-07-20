import numpy as np

def rmse(y, t):
	"""
	Returns the Root Mean Square Error of predicted values y
	from target values t.
	Args:
		y: numpy array of size (samples, features).
		t: numpy array of size (samples, features).
	Returns:
		float, the RMSE.
	"""
	N = y.shape[0]
	error_norms = np.linalg.norm(y - t, axis=1);
	return np.sqrt(np.dot(error_norms, error_norms)/N)
