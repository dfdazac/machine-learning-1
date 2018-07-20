# machine-learning-1

Code and notebooks based on the material for the ML1 course I took at the University of Amsterdam.

Install
-------

Requirements:

- Python
- Pipenv


```shell
# Create a new environment
$ pipenv shell
# Install the requirements
$ pipenv install --ignore-pipfile
# Close and open the shell again to update packages
$ exit
$ pipenv shell
# Explore the notebooks
$ jupyter notebook
```

Notebooks
---------

- [Maximum a posteriori signal detection](00-matched-filter.ipynb)
- [Linear models for regression](01-lr_ex.ipynb)
- [Predicting House Prices With Linear Regression](02-lr_housing.ipynb)
- [Bayesian Linear Regression](03-bayes_lr_ex.ipynb)
- [Dimensionality reduction and classification with Fisher's linear discriminant](04-fisher-example.ipynb)
- [Principal Component Analysis](05-pca_ex.ipynb)

Upcoming...

- Probabilistic Generative Models (Naive Bayes, etc) (classification)
- Logistic Regression
- Neural Networks
- Gaussian Processes
- Support Vector Machines
- Mixture Models
- Ensemble learning
