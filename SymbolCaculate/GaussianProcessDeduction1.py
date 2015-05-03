# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from sympy.abc import  *
from sympy import *
from sympy.stats import  *
init_printing()

# <codecell>

#n=4
#X= MatrixSymbol('X',n,1)
#Y= MatrixSymbol('Y',n,1)
#np.dot(X,Y)

# <headingcell level=2>

# squared exponent （SE）kernel

# <markdowncell>

# 定义符号

# <codecell>

sigma1,sigma2,sigman, deltaxx= symbols('sigma_1 sigma_2  sigma_n delta_{xx_1} ')
PAB = symbols('P_{AB}',cls =Function)

# <codecell>

K= symbols('K',cls =Function)
def K(x,y,n):
    X = MatrixSymbol(x,n,1)
    Y = MatrixSymbol(y,n,1)
    tmp=-1/2*(X-Y).T/M*(X-Y)
    return  sigma1**2*exp(tmp[0,0])
K("a","b",4)

# <codecell>

SigmaN=BlockMatrix([[K("X","X",1)+sigman**2, K("X","x",1)], [K("x","X",1), K("x","x",1)]])

# <markdowncell>

# 这里好像还是不太对，继续修改

# <headingcell level=2>

# 产生正态分布

# <codecell>

n=2
Mu = MatrixSymbol('Mu', n,1)
Sigma = MatrixSymbol("Sigma",n,n)
def normal(x,Mu,Sigma,n):
    X = MatrixSymbol(x,n,1)
    #normal =exp(-(X-Mu)*(X-Mu).T/(2*Sigma**2))/sqrt(2*pi)
    return -(X-Mu)*(X-Mu).T/(2*Sigma**2)
normal("X",Mu,Sigma,n)

# <codecell>

PA=norm(ZeroMatrix(1,n), sigman**2*Identity(n))

# <codecell>

PAB=norm(0, )

# <codecell>

sigmaj= BlockMatrix([[K(X,X)+d**2, K(X,x)],
         [ K(X,x), K(x,x)]])
sigmaj

# <codecell>

x= MatrixSymbol('x',1,2)

# <codecell>

normal =-x.doit()
#/sqrt(2*pi)

# <codecell>

#X= MatrixSymbol('X',n,1)
#x= MatrixSymbol('x',1,1)

# <codecell>

K(X,X)

# <codecell>

import scipy
scipy.__version__

# <markdowncell>

# 多变量正态分布

# <markdowncell>

# 这和普通的符号计算Matrixsymbol有什么区别？

# <headingcell level=2>

# [多变量正态分布](https://sympystats.wordpress.com/2011/07/19/multivariate-normal-random-variables/)

# <codecell>


