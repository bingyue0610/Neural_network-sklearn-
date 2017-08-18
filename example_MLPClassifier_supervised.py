#coding:utf-8
'''
class sklearn.neural_network.MLPClassifier(hidden_layer_sizes=(100, ), activation=’relu’, solver=’adam’,
alpha=0.0001, batch_size=’auto’, learning_rate=’constant’, learning_rate_init=0.001, power_t=0.5,
max_iter=200, shuffle=True, random_state=None, tol=0.0001, verbose=False, warm_start=False, momentum=0.9,
nesterovs_momentum=True, early_stopping=False, validation_fraction=0.1, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

parameters in detail:
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html#sklearn.neural_network.MLPClassifier

'''

'''
Currently, MLPClassifier supports only the Cross-Entropy loss function,
which allows probability estimates by running the predict_proba method.
'''


#example 1
from sklearn.neural_network import MLPClassifier
X = [[0., 0.], [1., 1.]]
y = [0, 1]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)
print clf.fit(X, y)
print clf.predict([[2., 2.], [-1., -2.]])
print [coef.shape for coef in clf.coefs_]
print clf.predict_proba([[2., 2.], [1., 2.]])

'''
fit(X, y):	Fit the model to data matrix X and target(s) y.
get_params([deep]):	Get parameters for this estimator.
predict(X):	Predict using the multi-layer perceptron classifier
predict_log_proba(X):	Return the log of probability estimates.
predict_proba(X):	Probability estimates.
score(X, y[, sample_weight]):	Returns the mean accuracy on the given test data and labels.
set_params(**params):	Set the parameters of this estimator.
'''

#example 2 multiclassifier
'''
Further, the model supports multi-label classification in which a sample can belong to more than one class.
For each class, the raw output passes through the logistic function. Values larger or equal to 0.5 are rounded to 1,
otherwise to 0. For a predicted output of a sample, the indices where the value is 1 represents
the assigned classes of that sample:
'''
X = [[0., 0.], [1., 1.]]
y = [[0, 1], [1, 1]]
clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
clf.fit(X, y)
print clf.predict([[1., 2.]])
print clf.predict([[0., 0.]])

