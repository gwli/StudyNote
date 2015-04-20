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

# squared exponential （SE） 核函数

# <markdowncell>

# 定义符号

# <codecell>

sigma1,sigma2, deltaxx= symbols('sigma_1 sigma_2 delta_{xx_1}')

# <codecell>

K= sigma1*exp(-(x-x1)**2/l)+sigma2*deltaxx
K

# <codecell>

K.subs(x,2)

# <codecell>


