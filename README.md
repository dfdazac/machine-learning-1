[![Build Status](https://travis-ci.org/dfdazac/machine-learning-1.svg?branch=master)](https://travis-ci.org/dfdazac/machine-learning-1)

# machine-learning-1

Code and notebooks based on the material for the ML1 course I took at the University of Amsterdam.

Install
-------

To get a reproducible environment, first install [Conda or Miniconda](https://conda.io/docs/user-guide/install/download.html) and then, create a new environment with the requirements with

```shell
$ conda env create -f environment.yml
```

Notebooks
---------

```shell
# Activate the environment
$ source activate machine-learning-1
# Explore the notebooks
$ jupyter notebook
```

- [Maximum a posteriori signal detection](00-matched-filter.ipynb)
- [Linear models for regression](01-lr_ex.ipynb)
- [Predicting House Prices With Linear Regression](02-lr_housing.ipynb)
- [Bayesian Linear Regression](03-bayes_lr_ex.ipynb)
- [Dimensionality reduction and classification with Fisher's linear discriminant](04-fisher-example.ipynb)
- [Principal Component Analysis](05-pca_ex.ipynb)
- [Neural networks with NumPy](06-neural-networks-numpy.ipynb)
- [Expectation Maximization](07-expectation-maximization.ipynb)

Upcoming...

- Gaussian Processes
- Support Vector Machines
