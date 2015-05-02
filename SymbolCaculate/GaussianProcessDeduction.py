# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sympy.abc import  *
from sympy.stats import  *
from sympy import *
from sympy.stats import E
import inspect
init_printing()

# <markdowncell>

# 函数f(x)的概率密度函数

# <codecell>

mu = Symbol('mu', real=True)
sigma = Symbol("sigma",positive=True)
f = Normal("f",mu,sigma)

# <markdowncell>

# 噪声的概率密度函数

# <codecell>

sigmaN = Symbol("sigma_N",positive=True)
xi = Normal("xi",0,sigmaN)
sigmaN

# <codecell>

y=f+xi

# <markdowncell>

# 显示期望

# <codecell>

E(y)

# <codecell>

std(y)

# <markdowncell>

# 加入新的测试值

# <codecell>

sigmaNew = Symbol("sigma_{New}",positive=True)
f1= Normal("f1",mu,sigmaNew)

# <codecell>

std(y+f1)

# <codecell>

n=4
mu = MatrixSymbol('mu', n, 1) 
Sigma = MatrixSymbol('Sigma',n,n,positive=True) 
X = Normal(mu, Sigma, 'X') 

# <headingcell level=2>

# 计算f的后验概率密度函数

# <headingcell level=2>

# squared exponential （SE） kernel

# <markdowncell>

# 定义符号

# <codecell>

sigma1,sigma2, deltaxx= symbols('sigma_1 sigma_2 delta_{xx_1}')

# <codecell>

def K(x,y):
    return  sigma1**2*exp(-1/2*(x-y)/M*(x-y))

# <codecell>

K(a,b)

# <markdowncell>

# 这里还需要修改？

# <markdowncell>

# 在高斯过程中，单张脸就具有期望和方差？

# <markdowncell>

# 产生正态分布

# <codecell>

from scipy.stats import norm

# <markdowncell>

# mu =0.5
# sigma = 1
# fp= norm(mu,sigma)

# <codecell>

from sympy.abc import  *

# <codecell>

mu = Symbol('mu', real=True)
sigma = Symbol("sigma",positive=True)

# <codecell>

def norm(mu,sigma):
    normal =exp(-(x-mu)**2/(2*sigma**2))/sqrt(2*pi)
    return normal

# <codecell>

norm(0, [[K(X,X)+d**2I, K(X,x)]
         [ K(X,x) K(x,x)]])

# <codecell>

2*I

# <codecell>

import sympy
sympy.__version__

# <markdowncell>

# 多变量正态分布

# <codecell>

from sympy.stats import  mu

# <codecell>


