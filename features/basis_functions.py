import numpy as np
from math import factorial
from itertools import combinations_with_replacement
from functools import reduce

def polynomial_basis(X, degree):
    """
    Calculates a polynomial function of the input X
    with the specified degree. For datasets of multiple variables,
    it calculates the multiset of monomials using all the features.
    For example, given a dataset with 2 features a and b, and an order
    3 polynomial, the resulting monomials are
    [1, a, b, a**2, a * b, b**2, a**3, a**2 * b, a * b**2, b**3].
    Args:
        X: numpy array of shape (samples, features).
        degree: int, degree of the polynomial
    Returns:
        Numpy array of shape (samples, (features + degree) C degree)
        where C denotes a combination (as in n choose k for nCk).
    """
    n_samples, n_features = X.shape

    # The number of monomials is (n + d) choose d
    n_monomials = int(factorial(n_features + degree)/(factorial(n_features)*factorial(degree)))
    features = np.ones((n_monomials, n_samples))
    col = 1
    x_T = X.T

    for deg in range(1, degree + 1):
        for combs in combinations_with_replacement(x_T, deg):
            features[col, :] = reduce(lambda x, y: x * y, combs)
            col += 1
    return features.T
