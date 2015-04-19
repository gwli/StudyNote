# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# [最大似然估计](http://nbviewer.ipython.org/github/rlabbe/Python-for-Signal-Processing/blob/master/Maximum_likelihood.ipynb)

# <codecell>

from __future__ import  division
from scipy.stats import bernoulli
import numpy as np

# <codecell>

p_true=1/2 # this is the value we will try to estimate from the observed data
fp=bernoulli(p_true)

def sample(n=10):
    'simulate coin flipping'
    return fp.rvs(n)# flip it n times

xs = sample(100) # generate some samples

# <codecell>

import sympy
from sympy.abc import x, z
p=sympy.symbols('p',positive=True)
sympy.init_printing()

# <codecell>

L=p**x*(1-p)**(1-x)
J=np.prod([L.subs(x,i) for i in xs]) # objective function to maximize
J

# <codecell>

logJ=sympy.expand_log(sympy.log(J))
sol=sympy.solve(sympy.diff(logJ,p),p)[0]

x=linspace(0,1,100)
plot(x,map(sympy.lambdify(p,logJ,'numpy'),x),sol,logJ.subs(p,sol),'o',
                                          p_true,logJ.subs(p,p_true),'s',)
xlabel('$p$',fontsize=18)
ylabel('Likelihood',fontsize=18)
title('Estimate not equal to true value',fontsize=18)

# <codecell>

L=p**x*(1-p)**(1-x)
J=np.prod([L.subs(x,i) for i in xs]) # objective function

