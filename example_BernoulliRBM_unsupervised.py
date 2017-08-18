#coding:utf-8

#example 1
import numpy as np
from sklearn.neural_network import BernoulliRBM
X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
model = BernoulliRBM(n_components=2)
print model.fit(X)
print model.get_params()
print model.score_samples([[0, 1, 1]])
print model.gibbs([[0, 1, 1]])

'''
fit(X[, y]):	Fit the model to the data X.
fit_transform(X[, y]):	Fit to data, then transform it.
get_params([deep]):	Get parameters for this estimator.
gibbs(v):	Perform one Gibbs sampling step.
partial_fit(X[, y]):	Fit the model to the data X which should contain a partial segment of the data.
score_samples(X):	Compute the pseudo-likelihood of X.
set_params(**params):	Set the parameters of this estimator.
transform(X):	Compute the hidden layer activation probabilities, P(h=1|v=X).
'''


