# Neural_network-sklearn-
尝试性的了解sklearn中的Neural Network Models
比较好用的是：MLPClassifier & MLPRegressor。只有MLPClassifier可以输出predict_probability.
BenoulliRBM适用于unsupervised learning。但是玩法还没有弄明白。特别是：gibbs(v)与score_samples的用法
另外，partial_fit的使用方式也没有弄清楚。

以上，重要调参包括：
hidden_layer_sizes : tuple, length = n_layers - 2, default (100,)
The ith element represents the number of neurons in the ith hidden layer.

activation : {‘identity’, ‘logistic’, ‘tanh’, ‘relu’}, default ‘relu’
Activation function for the hidden layer.
‘identity’, no-op activation, useful to implement linear bottleneck, returns f(x) = x
‘logistic’, the logistic sigmoid function, returns f(x) = 1 / (1 + exp(-x)).
‘tanh’, the hyperbolic tan function, returns f(x) = tanh(x).
‘relu’, the rectified linear unit function, returns f(x) = max(0, x)

solver : {‘lbfgs’, ‘sgd’, ‘adam’}, default ‘adam’
The solver for weight optimization.
‘lbfgs’ is an optimizer in the family of quasi-Newton methods.
‘sgd’ refers to stochastic gradient descent.
‘adam’ refers to a stochastic gradient-based optimizer proposed by Kingma, Diederik, and Jimmy Ba
Note: The default solver ‘adam’ works pretty well on relatively large datasets (with thousands of training samples or more) in terms of both training time and validation score. For small datasets, however, ‘lbfgs’ can converge faster and perform better.

alpha : float, optional, default 0.0001
L2 penalty (regularization term) parameter.

以下为其他可调参数，影响较小。
batch_size : int, optional, default ‘auto’
Size of minibatches for stochastic optimizers. If the solver is ‘lbfgs’, the classifier will not use minibatch. When set to “auto”, batch_size=min(200, n_samples)

learning_rate : {‘constant’, ‘invscaling’, ‘adaptive’}, default ‘constant’
Learning rate schedule for weight updates.
‘constant’ is a constant learning rate given by ‘learning_rate_init’.
‘invscaling’ gradually decreases the learning rate learning_rate_ at each time step ‘t’ using an inverse scaling exponent of ‘power_t’. effective_learning_rate = learning_rate_init / pow(t, power_t)
‘adaptive’ keeps the learning rate constant to ‘learning_rate_init’ as long as training loss keeps decreasing. Each time two consecutive epochs fail to decrease training loss by at least tol, or fail to increase validation score by at least tol if ‘early_stopping’ is on, the current learning rate is divided by 5.
Only used when solver='sgd'.

learning_rate_init : double, optional, default 0.001
The initial learning rate used. It controls the step-size in updating the weights. Only used when solver=’sgd’ or ‘adam’.

power_t : double, optional, default 0.5
The exponent for inverse scaling learning rate. It is used in updating effective learning rate when the learning_rate is set to ‘invscaling’. Only used when solver=’sgd’.

max_iter : int, optional, default 200
Maximum number of iterations. The solver iterates until convergence (determined by ‘tol’) or this number of iterations. For stochastic solvers (‘sgd’, ‘adam’), note that this determines the number of epochs (how many times each data point will be used), not the number of gradient steps.

shuffle : bool, optional, default True
Whether to shuffle samples in each iteration. Only used when solver=’sgd’ or ‘adam’.

random_state : int, RandomState instance or None, optional, default None
If int, random_state is the seed used by the random number generator; If RandomState instance, random_state is the random number generator; If None, the random number generator is the RandomState instance used by np.random.

tol : float, optional, default 1e-4
Tolerance for the optimization. When the loss or score is not improving by at least tol for two consecutive iterations, unless learning_rate is set to ‘adaptive’, convergence is considered to be reached and training stops.

verbose : bool, optional, default False
Whether to print progress messages to stdout.

warm_start : bool, optional, default False
When set to True, reuse the solution of the previous call to fit as initialization, otherwise, just erase the previous solution.

momentum : float, default 0.9
Momentum for gradient descent update. Should be between 0 and 1. Only used when solver=’sgd’.

nesterovs_momentum : boolean, default True
Whether to use Nesterov’s momentum. Only used when solver=’sgd’ and momentum > 0.

early_stopping : bool, default False
Whether to use early stopping to terminate training when validation score is not improving. If set to true, it will automatically set aside 10% of training data as validation and terminate training when validation score is not improving by at least tol for two consecutive epochs. Only effective when solver=’sgd’ or ‘adam’

validation_fraction : float, optional, default 0.1
The proportion of training data to set aside as validation set for early stopping. Must be between 0 and 1. Only used if early_stopping is True

beta_1 : float, optional, default 0.9
Exponential decay rate for estimates of first moment vector in adam, should be in [0, 1). Only used when solver=’adam’

beta_2 : float, optional, default 0.999
Exponential decay rate for estimates of second moment vector in adam, should be in [0, 1). Only used when solver=’adam’

epsilon : float, optional, default 1e-8
Value for numerical stability in adam. Only used when solver=’adam’
