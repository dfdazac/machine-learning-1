import numpy as np

def crossval_indices(N, k):
    """
    For each fold, returns indices of the training and validation folds
    to be used during cross-validation.
    Args:
        - N (int): the total number of samples
        - k (int): the number of folds
    Returns:
        - train_folds: numpy array of shape (N - N/k,) for training
        - valid_folds: numpy array of shape (N/k,) for validation
    """
    all_indices = np.arange(N, dtype=int)
    idx = [int(i) for i in np.floor(np.linspace(0,N,k+1))]
    train_folds = []
    valid_folds = []
    for fold in range(k):
        valid_indices = all_indices[idx[fold]:idx[fold+1]]
        valid_folds.append(valid_indices)
        train_folds.append(np.setdiff1d(all_indices, valid_indices))
    return train_folds, valid_folds

def split_train_test(X, Y, split=0.7):
    # TODO: Add docstring
    split_idx = int(len(X) * split)
    X_train, Y_train = X[:split_idx], Y[:split_idx]
    X_test, Y_test = X[split_idx:], Y[split_idx:]
    return X_train, Y_train, X_test, Y_test
