# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>


# <codecell>

from sympy.abc import  *
from sympy import *
from sympy.stats import  *
init_printing()

# <headingcell level=2>

# squared exponent （SE）kernel

# <markdowncell>

# 定义符号

# <codecell>

sigma1,sigma2,sigman, deltaxx= symbols('sigma_1 sigma_2  sigma_n delta_{xx_1} ')
PAB = symbols('P_{AB}',cls =Function)

# <codecell>

n=4
K= symbols('K',cls =Function)
X= MatrixSymbol('X',n,1)
Y= MatrixSymbol('Y',n,1)
def K(x,y):
    tmp=-1/2*(x-y).T/M*(x-y)
    return  sigma1**2*exp(tmp[0,0])

# <markdowncell>

# 这里下面无法表达任意的，主要还要修改 搜索  def matrix operation

# <headingcell level=2>

# 产生正态分布

# <codecell>

mu = Symbol('mu', real=True)
sigma = Symbol("sigma",positive=True)

# <codecell>

def norm(mu,sigma):
    normal =exp(-(x-mu)**2/(2*sigma**2))/sqrt(2*pi)
    return normal

# <codecell>

norm(mu,sigma)

# <codecell>

PA=norm(ZeroMatrix(1,n), sigman**2*Identity(n))

# <codecell>

PAB=norm(ZeroMatrix(n+1,n+1), BlockMatrix([K(X,X)+d**2*dentity(n), K(X,x)],
         [ K(X,x) K(x,x)]))

# <codecell>

X= MatrixSymbol('X',n,1)
x= MatrixSymbol('x',1,1)

# <codecell>

Kernel(X,X)

# <codecell>

K(X,X)

# <codecell>

BlockMatrix([[K(X,X)+sigman**2, K(X,x)], [K(x,X), K(x,x)]])

# <markdowncell>

# 多变量正态分布

# <markdowncell>

# 这和普通的符号计算Matrixsymbol有什么区别？

