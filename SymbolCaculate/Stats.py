# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# 统计知识

# <headingcell level=2>

# [正态分布可视化](http://www.cnblogs.com/vamei/p/3199522.html)

# <codecell>

from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

# <codecell>

# By Vamei
rv1 = norm(loc=0, scale = 1)
rv2 = norm(loc=2, scale = 1)
rv3 = norm(loc=0, scale = 2)
x = np.linspace(-5, 5, 200)

plt.plot(x, rv1.pdf(x), label="N(0,1)")
plt.plot(x, rv2.pdf(x), label="N(2,1)")
plt.plot(x, rv3.pdf(x), label="N(0,2)")
plt.legend()

plt.xlim([-5, 5])
plt.title("normal distribution")
plt.xlabel("RV")
plt.ylabel("f(x)")

plt.show()

# <markdowncell>

# 这里的颜色是自动画出来的吗？

# <markdowncell>

#                 这个说明方差越大，分布覆盖的范围越大。

# <headingcell level=2>

# 指数分布

# <markdowncell>

# 概率密度函数：
# 
# $$f(x) = \left\{ \begin{array}{rcl} \lambda e^{-\lambda x} & if & x \ge 0 \\ 0 & if & x < 0 \end{array} \right. $$

# <headingcell level=2>


# <codecell>

from scipy.stats import expon
import numpy as np
import matplotlib.pyplot as plt

# <codecell>

rve1 = expon(scale = 5)
rve2 = expon(scale = 10)
rve3 = expon(scale = 0)

x = np.linspace(0, 20, 100)
plt.plot(x, rve1.pdf(x))
plt.plot(x, rve2.pdf(x))
plt.plot(x, rve3.pdf(x))
plt.xlim([0, 15])
plt.title("exponential distribution")
plt.xlabel("RV")
plt.ylabel("f(x)")
plt.show()

# <headingcell level=2>

# [Gamma 分布](http://docs.scipy.org/doc/scipy-0.15.1/reference/generated/scipy.stats.gamma.html)

# <headingcell level=3>

# Gamma分布概率密度函数

# <markdowncell>

# $Gamma(t|\alpha,\beta)=\frac{\beta^\alpha  t^{(\alpha-1)}  e^{(-\beta t)} }{\Gamma(\alpha)}$

# <codecell>

from scipy.stats import  gamma
import matplotlib.pylab as plt
from sympy.abc import a,x
from sympy import init_printing
init_printing(use_unicode='True')
import numpy as np

# <codecell>

rvg1 = gamma(3., loc = 0., scale = 2.)
rvg2 = gamma(0., loc = 0., scale = 1.)
rvg3 = gamma(0., loc = 1., scale = 1.)
x = np.linspace(-10,10,100)
plt.plot(x,rvg1.pdf(x),label='Gamma(3,0,2)')
plt.plot(x,rvg2.pdf(x),label='Gamma(0,0,10)')
plt.plot(x,rvg3.pdf(x),label='Gamma(0,1,1)')
plt.legend()
plt.xlim(-5,5)
plt.title('Gamma distribution')
plt.xlabel('RV')
plt.ylabel('f(x)')

# <headingcell level=3>

# [T分布](http://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.stats.t.html)

# <codecell>

fig,ax = plt.subplots(1,1)

# <codecell>

df =2.74335149908
mean,var,skew,kurt = t.stats(df,moments ='mvsk') 

# <codecell>

x = np.linspace(t.ppf(0.01,df),t.ppf(0.99,df),100)

# <codecell>

ax.plot(x,t.pdf(x,df), 'r-', lw=5, alpha=0.6, label='t pdf')

# <codecell>

import math
import random

# <headingcell level=2>

# [t分布产生](http://www.johndcook.com/python_student_t_rng.html)

# <codecell>

def student_t(nu):
    x =random.gauss(0,1)
    y = 2.0* random.gammavariate(0.5*nu,2.0)
    return  x/(math.sqrt(y/nu))

# <codecell>

student_t(100)

# <headingcell level=3>

# 多变量正态分布

# <codecell>

import matplotlib.pyplot as plt
import numpy as np
import matplotlib.mlab as mlab

# <codecell>

delta =0.025
x = np.arange(-2.0,3.0,delta)
y = np.arange(-2.0,2.0,delta)
X,Y = np.meshgrid(x,y)
Z1= mlab.bivariate_normal(X,Y,1.0, 1.0, 0.0, 0.0)
Z2= mlab.bivariate_normal(X, Y, 1.5, 0.5, 1, 1)

# <markdowncell>

# (X,Y,1.0, 1.0, 0.0, 0.0) 多变量正态分布中的这里代表什么?

# <codecell>

plt.figure()
CS = plt.contour(X,Y,Z2)

# <codecell>


