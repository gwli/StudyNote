# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import print_function

from sympy import Symbol, symbols, lambdify
from sympy import sqrt as SQRT
from sympy.stats import Normal, E, variance, skewness
from math import sqrt
from sympy import  init_printing

# <codecell>

class ProductNormalNormal():
    def __init__(self):
        mu1 = Symbol('mu1', positive=True, real=True, bounded=True)
        s1 = Symbol('s1', positive=True, real=True, bounded=True)
        mu2 = Symbol('mu2', positive=True, real=True, bounded=True)
        s2 = Symbol('s2', positive=True, real=True, bounded=True)

        N1 = Normal('N1', mu1, s1)
        N2 = Normal('N2', mu2, s2)
        NN = N1 * N2

        self.MeanNN = E(NN)
        self.VarNN = variance(NN)
        self.StdevNN = SQRT(self.VarNN)
        self.SkewNN = skewness(NN)

        self.meanNN = lambdify([mu1, s1, mu2, s2], self.MeanNN)
        self.varNN = lambdify([mu1, s1, mu2, s2], self.VarNN)
        self.stdevNN = lambdify([mu1, s1, mu2, s2], self.StdevNN)
        self.skewNN = lambdify([mu1, s1, mu2, s2], self.SkewNN)

# <codecell>

class ProductOfNormals():
    def __init__(self, n):
        mu = symbols('mu0:%d' % n, positive=True, real=True, bounded=True)
        s = symbols('s0:%d' % n, positive=True, real=True, bounded=True)
        N = []
        for i in range(n):
            N.append(Normal('N%d' % i, mu[i], s[i]))

        NN = N[-1]
        for i in range(n - 1):
            NN *= N[i]

        self.DistributionNN = NN
        self.MeanNN = E(NN)
        self.VarNN = variance(NN)
        self.StdevNN = SQRT(self.VarNN)
        self.SkewNN = skewness(NN)

        self.meanNN = lambdify([mu, s], self.MeanNN)
        self.varNN = lambdify([mu, s], self.VarNN)
        self.stdevNN = lambdify([mu, s], self.StdevNN)
        self.skewNN = lambdify([mu, s], self.SkewNN)

    def meanOfNs(self, mu, s):
        mus = mu + s
        return self.meanNN(*mus)

    def varOfNs(self, mu, s):
        mus = mu + s
        return self.varNN(*mus)

    def stdevOfNs(self, mu, s):
        mus = mu + s
        return self.stdevNN(*mus)

    def skewOfNs(self, mu, s):
        mus = mu + s
        return self.skewNN(*mus)

    def densityOfNs(self, x, mu, s):
        mus = mu + s
        return self.densityNN(*mus)(x)

# <codecell>

if __name__ == '__main__':
    from sympy import pprint
    n = 2
    mu0 = 1.0
    s0 = 0.05
    mu1 = 1.0
    s1 = 0.15

    Nn = ProductOfNormals(n)
    print('Sympy formula product of %d N(mu,s)', n)
    print('Mean')
    pprint(Nn.MeanNN)
    print('\nVariance')
    pprint(Nn.VarNN)
    print('\nSkewness')
    pprint(Nn.SkewNN)

    s0n = s0 / sqrt(n - 1)
    print('\ns0:     ', s0)
    print('s0n:    ', s0n)
    mu = [mu1]
    s = [s1]
    for i in range(n - 1):
        mu.insert(0, mu0)
        s.insert(0, s0)

    print('\nMean:   ', Nn.meanOfNs(mu, s))
    print('Var:    ', Nn.varOfNs(mu, s))
    print('Stdev:  ', Nn.stdevOfNs(mu, s))
    print('Skew:   ', Nn.skewOfNs(mu, s))

    NN = ProductNormalNormal()
    print('\nSympy formula NN(mu,s)*NN(mu,s)')
    print('Mean')
    pprint(Nn.MeanNN)
    print('\nVariance')
    pprint(Nn.VarNN)
    print('\nSkewness')
    pprint(Nn.SkewNN)

    print('\nMean:   ', NN.meanNN(mu0, s0, mu1, s1))
    print('Var:    ', NN.varNN(mu0, s0, mu1, s1))
    print('Stdev:  ', NN.stdevNN(mu0, s0, mu1, s1))
    print('Skew:   ', NN.skewNN(mu0, s0, mu1, s1))

# <codecell>


