import numpy as np

class BernoulliMixture:
    """A Bernoulli mixture model.
    Args:
        dimensions (int): dimensionality of the data
        n_components (int): the number of components
        mu (array): shape (K, D), values to initialize distribution means
        pi (array): shape (K,), values to initialize mixing coefficients
    """
    def __init__(self, dimensions, n_components, mu=None, pi=None,
        verbose=False):
        # Initialize distribution parameters
        if mu is None:
            # Random probabilities for each component
            self.mu = np.random.uniform(low=0.25, high=0.75, size=(n_components, D))
        else:
            # Ensure that mu does not contain problematic values
            self.mu = np.clip(mu, 1e-12, 1 - 1e-12)

        if pi is None:
            # Equally likely components
            self.pi = np.ones(n_components) / n_components
        else:
            self.pi = pi

        self.verbose = verbose

    def fit(X):
        """Find the parameters of the mixture model given the data. It uses
        Expectation Maximization to find maximum likelihood estimates of the
        component means and mixing coefficients.
        Args:
            X (N, D): data
        """
        if X.shape[1] != mu.shape[1]:
            raise ValueError(('Invalid data dimensions, expected {:d},'
                'got {:d}').format(mu.shape[1], X.shape[1]))

        prev_nll = 0
        for i in range(max_iter):
            posterior, nll = self._e_step(X, self.mu, self.pi)
            self.mu, self.pi = self._m_step(X, posterior)

            # Check convergence on the negative log-likelihood
            if np.abs(prev_nll - nll) < 1:
                if verbose:
                    print('Terminating early')
                break
            prev_nll = nll

            if self.verbose:
                print('\r{:d}/{:d}  NLL: {:.3f}'.format(i+1, max_iter, nll),
                    end='', flush=True)

    def _e_step(X, mu, pi):
        """ Performs the E step using the given arrays, calculating the
        posterior probabilities of each component, for each data point.
        N is the number of data points, D the dimensionality of
        each data point, K the number of components of the mixture.
        Args:
            X (N, D): data
            mu (K, D): component probabilities
            pi (K,): mixing coefficients
        Returns:
            posteriors: array, shape (N, K), the posterior probabilities
            nll: float, the negative log-likelihood of the data given mu and pi
        """
        # Expand the dimensions of mu to broadcast operations with X
        mu_ex = mu[:, np.newaxis]
        # The shape of mu_ex is (K, 1, D), so after broadcasting
        # the operations with X, the result will have shape (K, N, D)

        # Probability of the data given the components (Bishop, eq. 9.48)
        # (actually we use log-probabilities and later the log-sum-exp trick
        # for stability). The result has shape (K, N).
        log_k_probs = np.sum(X * np.log(mu_ex) + (1 - X) * np.log(1 - mu_ex), axis=2)

        # Unnormalized posterior (Bishop, numerator of eq. 9.56)
        # Result is (K, N)
        un_log_posterior = np.log(pi[:, np.newaxis]) + log_k_probs

        # Constants for the log-sum-exp trick, with shape (N,)
        a = np.max(un_log_posterior, axis=0)

        # Normalize posterior (to finish eq. 9.56) using log-sum-exp trick
        # shape: (N,)
        normalizer = a + np.log(np.sum(np.exp(un_log_posterior - a), axis=0))
        # shape: (K, N)
        log_posteriors = un_log_posterior - normalizer

        # Calculate negative log-likelihood for convergence check
        nll = -np.sum(normalizer)

        # log_posteriors has shape (K, N) so we transpose as needed
        return np.exp(log_posteriors.T), nll

    def _m_step(X, posterior):
        """Performs the M step given the data and posterior probabilities,
        calculating the parameters that maximize the likelihood function.
        N is the number of data points, D the dimensionality of
        each data point, K the number of components of the mixture.
        Args:
            X (N, D): data
            posterior (N, K): posterior probabilities of each component, for
                each data point
        Returns:
            mu (K, D): component probabilities
            pi (K,): mixing coefficients
        """
        # Effective points assigned to component k (Bishop, eq. 9.57)
        N_k = np.sum(posterior, axis=0)

        # Maximizing mu (Bishop, eqs. 9.58 and 9.59)
        mu = (posterior.T @ X) / N_k[:, np.newaxis]
        # Control extreme values of mu
        np.clip(mu, 1e-12, 1 - 1e-12, out=mu)

        # Maximizing pi (Bishop, eq. 9.60)
        pi = N_k / X.shape[0]

        return mu, pi

    def predict(X):
        pass
