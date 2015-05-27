# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # 使用[gaussian process 实现分类](http://scikit-learn.org/stable/auto_examples/gaussian_process/plot_gp_probabilistic_classification_after_regression.html)

# <codecell>

from __future__ import print_function
from scipy import linalg, optimize
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.utils import check_random_state, check_array, check_X_y
from sklearn.utils.validation import check_is_fitted
from sklearn.gaussian_process import regression_models as regression
from sklearn.gaussian_process import correlation_models as correlation

# <codecell>

inspect.getsourcelines(check_is_fitted)

# <markdowncell>

# 标准分布函数

# <codecell>

import numpy as np
from scipy import stats
#from sklearn.gaussian_process import GaussianProcess
from matplotlib import  pyplot as plt
from matplotlib import  cm
from sympy import *
from sympy.abc import *
import inspect
import matplotlib.pyplot as plt

# <markdowncell>

# 这里指的是什么？pdf, cdf ,ppf?

# <codecell>

phi = stats.distributions.norm().pdf
PHI = stats.distributions.norm().cdf
PHIinv = stats.distributions.norm().ppf

# <codecell>

lim =8

# <codecell>

def g(x):
    return 5.-x[:,1]-.5*x[:,0]**2

# <markdowncell>

# def 的函数怎样显示出来？

# <codecell>

X = np.array([[-4.61611719, -6.00099547],
              [4.10469096, 5.32782448],
              [0.00000000, -0.50000000],
              [-6.17289014, -4.6984743],
              [1.3109306, -6.93271427],
              [-5.03823144, 3.10584743],
              [-2.87600388, 6.74310541],
              [5.21301203, 4.26386883]])
X= np.random.randn(8,2)

# <codecell>

def g(x):
    return 5.-x[:,1]-.5*x[:,0]**2

# <markdowncell>

# 这里怎样拟合二维数据？ 这能只能处理两个特性，更多的特性拟合不出来。

# <codecell>

y =g(X)
y.shape

# <codecell>

_regression_types = {
        'constant': regression.constant,
        'linear': regression.linear,
        'quadratic': regression.quadratic}

_correlation_types = {
        'absolute_exponential': correlation.absolute_exponential,
        'squared_exponential': correlation.squared_exponential,
        'generalized_exponential': correlation.generalized_exponential,
        'cubic': correlation.cubic,
        'linear': correlation.linear}

_optimizer_types = [
        'fmin_cobyla',
        'Welch']

# <codecell>

def reduced_likelihood_function(theta=None):
       """
       This function determines the BLUP parameters and evaluates the reduced
       likelihood function for the given autocorrelation parameters theta.
       Maximizing this function wrt the autocorrelation parameters theta is
       equivalent to maximizing the likelihood of the assumed joint Gaussian
       distribution of the observations y evaluated onto the design of
       experiments X.
       Parameters
       ----------
       theta : array_like, optional
           An array containing the autocorrelation parameters at which the
           Gaussian Process model parameters should be determined.
           Default uses the built-in autocorrelation parameters
           (ie ``theta = theta_``).
       Returns
       -------
       reduced_likelihood_function_value : double
           The value of the reduced likelihood function associated to the
           given autocorrelation parameters theta.
       par : dict
           A dictionary containing the requested Gaussian Process model
           parameters:
               sigma2
                       Gaussian Process variance.
               beta
                       Generalized least-squares regression weights for
                       Universal Kriging or given beta0 for Ordinary
                       Kriging.
               gamma
                       Gaussian Process weights.
               C
                       Cholesky decomposition of the correlation matrix [R].
               Ft
                       Solution of the linear equation system : [R] x Ft = F
               G
                       QR decomposition of the matrix Ft.
       """
       check_is_fitted(GaussianProcess,'X')

       if theta is None:
           # Use built-in autocorrelation parameters
           theta = theta_

       # Initialize output
       reduced_likelihood_function_value = - np.inf
       par = {}

       # Retrieve data
       n_samples = X.shape[0]
       D = D
       ij = ij
       F = F

       if D is None:
           # Light storage mode (need to recompute D, ij and F)
           D, ij = l1_cross_distances(X)
           if (np.min(np.sum(D, axis=1)) == 0.
                   and corr != correlation.pure_nugget):
               raise Exception("Multiple X are not allowed")
           F = regr(X)

       # Set up R
       r = corr(theta, D)
       R = np.eye(n_samples) * (1. + nugget)
       R[ij[:, 0], ij[:, 1]] = r
       R[ij[:, 1], ij[:, 0]] = r

       # Cholesky decomposition of R
       try:
           C = linalg.cholesky(R, lower=True)
       except linalg.LinAlgError:
           return reduced_likelihood_function_value, par

       # Get generalized least squares solution
       Ft = linalg.solve_triangular(C, F, lower=True)
       try:
           Q, G = linalg.qr(Ft, econ=True)
       except:
           #/usr/lib/python2.6/dist-packages/scipy/linalg/decomp.py:1177:
           # DeprecationWarning: qr econ argument will be removed after scipy
           # 0.7. The economy transform will then be available through the
           # mode='economic' argument.
           Q, G = linalg.qr(Ft, mode='economic')
           pass

       sv = linalg.svd(G, compute_uv=False)
       rcondG = sv[-1] / sv[0]
       if rcondG < 1e-10:
           # Check F
           sv = linalg.svd(F, compute_uv=False)
           condF = sv[0] / sv[-1]
           if condF > 1e15:
               raise Exception("F is too ill conditioned. Poor combination "
                               "of regression model and observations.")
           else:
               # Ft is too ill conditioned, get out (try different theta)
               return reduced_likelihood_function_value, par

       Yt = linalg.solve_triangular(C, y, lower=True)
       if beta0 is None:
           # Universal Kriging
           beta = linalg.solve_triangular(G, np.dot(Q.T, Yt))
       else:
           # Ordinary Kriging
           beta = np.array(beta0)

       rho = Yt - np.dot(Ft, beta)
       sigma2 = (rho ** 2.).sum(axis=0) / n_samples
       # The determinant of R is equal to the squared product of the diagonal
       # elements of its Cholesky decomposition C
       detR = (np.diag(C) ** (2. / n_samples)).prod()

       # Compute/Organize output
       reduced_likelihood_function_value = - sigma2.sum() * detR
       par['sigma2'] = sigma2 * y_std ** 2.
       par['beta'] = beta
       par['gamma'] = linalg.solve_triangular(C.T, rho)
       par['C'] = C
       par['Ft'] = Ft
       par['G'] = G

       return reduced_likelihood_function_value, par

# <codecell>

def l1_cross_distances(X):
    """
    Computes the nonzero componentwise L1 cross-distances between the vectors
    in X.
    Parameters
    ----------
    X: array_like
        An array with shape (n_samples, n_features)
    Returns
    -------
    D: array with shape (n_samples * (n_samples - 1) / 2, n_features)
        The array of componentwise L1 cross-distances.
    ij: arrays with shape (n_samples * (n_samples - 1) / 2, 2)
        The indices i and j of the vectors in X associated to the cross-
        distances in D: D[k] = np.abs(X[ij[k, 0]] - Y[ij[k, 1]]).
    """
    X = check_array(X)
    n_samples, n_features = X.shape
    n_nonzero_cross_dist = n_samples * (n_samples - 1) / 2
    ij = np.zeros((n_nonzero_cross_dist, 2), dtype=np.int)
    D = np.zeros((n_nonzero_cross_dist, n_features))
    ll_1 = 0
    for k in range(n_samples - 1):
        ll_0 = ll_1
        ll_1 = ll_0 + n_samples - k - 1
        ij[ll_0:ll_1, 0] = k
        ij[ll_0:ll_1, 1] = np.arange(k + 1, n_samples)
        D[ll_0:ll_1] = np.abs(X[k] - X[(k + 1):n_samples])

    return D, ij

# <codecell>

beta0=None
storage_mode='full'
verbose=False
theta0=1e-1
thetaL=None
thetaU=None
optimizer='fmin_cobyla'
random_start=1
normalize=True
nugget=10. * MACHINE_EPSILON
random_state=None

# <codecell>

def fit(X, y):
    """
    The Gaussian Process model fitting method.
    Parameters
    ----------
    X : double array_like
        An array with shape (n_samples, n_features) with the input at which
        observations were made.
    y : double array_like
        An array with shape (n_samples, ) or shape (n_samples, n_targets)
        with the observations of the output to be predicted.
    Returns
    -------
    gp : self
        A fitted Gaussian Process model object awaiting data to perform
        predictions.
    """
    random_state=None
    random_state = check_random_state(random_state)

    # Force data to 2D numpy.array
    X, y = check_X_y(X, y, multi_output=True, y_numeric=True)
    y_ndim_ = y.ndim
    if y.ndim == 1:
        y = y[:, np.newaxis]

    # Check shapes of DOE & observations
    n_samples, n_features = X.shape
    _, n_targets = y.shape


    # Normalize data or don't
    if normalize:
        X_mean = np.mean(X, axis=0)
        X_std = np.std(X, axis=0)
        y_mean = np.mean(y, axis=0)
        y_std = np.std(y, axis=0)
        X_std[X_std == 0.] = 1.
        y_std[y_std == 0.] = 1.
        # center and scale X if necessary
        X = (X - X_mean) / X_std
        y = (y - y_mean) / y_std
    else:
        X_mean = np.zeros(1)
        X_std = np.ones(1)
        y_mean = np.zeros(1)
        y_std = np.ones(1)

    # Calculate matrix of distances D between samples
    D, ij = l1_cross_distances(X)
    if (np.min(np.sum(D, axis=1)) == 0.
            and corr != correlation.pure_nugget):
        raise Exception("Multiple input features cannot have the same"
                        " target value.")

    # Regression matrix and parameters
    regr=regression.constant
    corr=correlation.squared_exponential
    F = regr(X)
    n_samples_F = F.shape[0]
    if F.ndim > 1:
        p = F.shape[1]
    else:
        p = 1
    if n_samples_F != n_samples:
        raise Exception("Number of rows in F and X do not match. Most "
                        "likely something is going wrong with the "
                        "regression model.")
    if p > n_samples_F:
        raise Exception(("Ordinary least squares problem is undetermined "
                         "n_samples=%d must be greater than the "
                         "regression model size p=%d.") % (n_samples, p))
    if beta0 is not None:
        if beta0.shape[0] != p:
            raise Exception("Shapes of beta0 and F do not match.")

    # Set attributes
    X = X
    y = y
    D = D
    ij = ij
    F = F
    X_mean, X_std = X_mean, X_std
    y_mean, y_std = y_mean, y_std

    # Determine Gaussian Process model parameters
    if thetaL is not None and thetaU is not None:
        # Maximum Likelihood Estimation of the parameters
        if verbose:
            print("Performing Maximum Likelihood Estimation of the "
                  "autocorrelation parameters...")
        theta_, reduced_likelihood_function_value_, par = _arg_max_reduced_likelihood_function()
        if np.isinf(reduced_likelihood_function_value_):
            raise Exception("Bad parameter region. "
                            "Try increasing upper bound")

    else:
        # Given parameters
        if verbose:
            print("Given autocorrelation parameters. "
                  "Computing Gaussian Process model parameters...")
        theta_ = theta0
        reduced_likelihood_function_value_, par = reduced_likelihood_function()
        if np.isinf(reduced_likelihood_function_value_):
            raise Exception("Bad point. Try increasing theta0.")

    beta = par['beta']
    gamma = par['gamma']
    sigma2 = par['sigma2']
    C = par['C']
    Ft = par['Ft']
    G = par['G']

    if storage_mode == 'light':
        # Delete heavy data (it will be computed again if required)
        # (it is required only when MSE is wanted in predict)
        if verbose:
            print("Light storage mode specified. "
                  "Flushing autocorrelation matrix...")
        D = None
        ij = None
        F = None
        C = None
        Ft = None
        G = None

    return self

# <codecell>

fit(X,y)

# <codecell>

hasattr(GaussianProcess,'fit')

# <codecell>

type(GaussianProcess).__name__

# <codecell>

gp =GaussianProcess(theta0=5e-1)
gp.fit(X,y)

# <codecell>

y

# <codecell>

res = 50
x1,x2 = np.meshgrid(np.linspace(-lim,lim,res),
                     np.linspace(-lim,lim,res))

# <codecell>

xx = np.vstack([x1.reshape(x1.size),x2.reshape(x2.size)]).T
x1.reshape(x1.size).shape

# <codecell>

y_true = g(xx)
y_pred, MSE = gp.predict(xx,eval_MSE=True)

# <codecell>

sigma = np.sqrt(MSE)
y_true = y_true.reshape((res,res))
y_pred = y_pred.reshape((res,res))
sigma = y_pred.reshape((res,res))
k = PHIinv (.975)

# <codecell>

fig = plt.figure(1)
ax = fig.add_subplot(111)
ax.axes.set_aspect('equal')
plt.xticks([])
plt.yticks([])
ax.set_xticklabels([])
ax.set_yticklabels([])
plt.xlabel('$x_1$')
plt.ylabel('$x_2$')
cax = plt.imshow(np.flipud(PHI(- y_pred / sigma)), cmap=cm.gray_r, alpha=0.8,
                extent=(- lim, lim, - lim, lim))
norm = plt.matplotlib.colors.Normalize(vmin= 0.,vmax=0.9)
cb = plt.colorbar(cax,ticks = [0., 0.2, 0.4, 0.6, 0.8, 1.],norm= norm)
cb.set_label('${\\rm \mathbb{P}}\left[\widehat{G}(\mathbf{x}) \leq 0\\right]$')
plt.plot(X[y <= 0, 0], X[y <= 0, 1], 'r.', markersize=12)
plt.plot(X[y> 0, 0], X[y > 0, 1], 'b.', markersize=12)
cs = plt.contour(x1,x2,y_true,[0.],colors='r',linestyles = 'dashdot')
cs = plt.contour(x1,x2,PHI(-y_pred/sigma),[0.025],colors='b',linestyles ='solid')
plt.clabel(cs,fontsize =11)
cs = plt.contour(x1,x2,PHI(-y_pred/sigma),[0.5],colors='g',linestyles ='dashed')
plt.clabel(cs,fontsize =11)
cs =plt.contour(x1,x2,PHI(-y_pred/sigma),[0.975],colors='k',linestyles ='solid')
plt.clabel(cs,fontsize =11)
plt.show()

# <markdowncell>

# plt.contour 是轮廓，有什么意义？

# <codecell>

import inspect

# <codecell>

inspect(GaussianProcess)

# <codecell>


# <codecell>


