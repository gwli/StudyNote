# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # gaussian process

# <codecell>

import numpy as np
from sklearn import gaussian_process

# <codecell>

def f(x):
    return x* np.sin(x)

# <codecell>

X = np.atleast_2d(np.linspace(-10,10,50)).T
y = f(X).ravel()
x = np.atleast_2d(np.linspace(-10, 10, 15)).T

# <markdowncell>

# 这里X数据不能超过100？

# <codecell>

print X.shape
print y.shape

# <codecell>

gp = gaussian_process.GaussianProcess(theta0 = 1e-2, thetaL = 1e-4, thetaU =1e-1)
gp.fit(X,y)

# <markdowncell>

# y.ravel() 表示什么意思？

# <codecell>

y_pred, sigma2_pred = gp.predict(x,eval_MSE= True)

# <markdowncell>

# 显示结果

# <codecell>

plt.plot(X,y,'r--',label='test data')
plt.plot(x,y_pred,'g*--',label='predicted result')
plt.plot(x,f(x),'y^--',label='real function')
plt.legend()

# <markdowncell>

# np.atleast_2d 是什么用途？

# <markdowncell>

# 说明这个函数是有效的

# <codecell>


