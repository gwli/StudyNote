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
SigmaN

# <codecell>


# <headingcell level=2>

# 定义多元正态分布

# <markdowncell>

# 这里好像还是不太对，继续修改

# <markdowncell>

# 这里的shape 为什么无法使用？

# <codecell>

n=2
Mu = MatrixSymbol('Mu', n,1)
Sigma = MatrixSymbol("Sigma",n,n)
def normal(x,Mu,Sigma,n):
    X = MatrixSymbol(x,n,1)
    return -(X-Mu)*(X-Mu).T/(2*Sigma**2)
normal("X",Mu,Sigma,n)

# <markdowncell>

# Sigma.shape 是可以的，但是SigmaN.shape是不行的，说民两个不是同一种类型的？

# <codecell>

PA=normal('X',ZeroMatrix(n,1), sigman**2*Identity(n),1)

# <codecell>

sigmaj= BlockMatrix([[K(X,X)+d**2, K(X,x)],
         [ K(X,x), K(x,x)]])
sigmaj

# <codecell>

x= MatrixSymbol('x',1,2)

# <codecell>

#X= MatrixSymbol('X',n,1)
#x= MatrixSymbol('x',1,1)

# <codecell>

K(X,X)

# <markdowncell>

# 多变量正态分布

# <markdowncell>

# 这和普通的符号计算Matrixsymbol有什么区别？

# <headingcell level=2>

# [多变量正态分布](https://sympystats.wordpress.com/2011/07/19/multivariate-normal-random-variables/)

# <codecell>


