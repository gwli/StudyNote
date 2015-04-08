# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# 怎样生成 目录

# <markdowncell>

# # 符号推倒

# <markdowncell>

# ## 数值计算

# <markdowncell>

# ## 符号推倒

# <markdowncell>

# ### 向量计算

# <markdowncell>

# ### 公式可视化

# <markdowncell>

# #### 使用LaTeX命令  latex格式的

# <codecell>

from sympy  import  *
from sympy.abc import *

# <codecell>

def g(x):
    return x**2+x*2

# <codecell>

latex(g(x))

# <markdowncell>

# ##### 使用sympy 可视化

# <codecell>

from sympy.plotting import  plot

# <codecell>

plot(g(x),(x,-3,3))

# <markdowncell>

# 这个和matplot 中的有什么区别，我想这个只是为了显示,才需要数据段。

# <markdowncell>

# #### 使用pprint

# <codecell>

print g(x)

# <codecell>

pprint(g(x))

# <markdowncell>

# ####使用Ipython display，只是显示出来

# <codecell>

from IPython.display import display
display(g(x))

# <markdowncell>

# ### 矩阵推倒

# <codecell>

from sympy  import  *
from sympy.abc import *

# <codecell>

u,v,x,y,A = symbols('u v x y A')
G,H,F,N,g,h,f,n= symbols('G H F N g h f n',cls=Function)
Eq_5_6_2=Eq(H(u,v),G(u,v)/A)
Eq_5_5_17=Eq(G(u,v),H(u,v)*F(u,v)+N(u,v))

# <codecell>

ft=integrate(G(x)*E**(-I*w*x),x)
ft
s,n,f =symbols('s n f')ft.expand().combsimp()
ban=ft.expand()
ban.collect(s)
ban.coeff(s,2)

# <markdowncell>

# #### 产生矩阵符号

# <codecell>

I,J,K=6,8,7 
F=3
A=MatrixSymbol('A',I,F)
B=MatrixSymbol('B',J,F)
C=MatrixSymbol('C',K,F)
X_i=B*diag(A[0,0],A[0,1],A[0,2])*transpose(C)
X_i.

# <markdowncell>

# #### 加入旋转矩阵

# <codecell>

from sympy import sin, cos, Matrix
from sympy.abc import rho, phi

# <codecell>

X = Matrix([rho*cos(phi), rho*sin(phi), rho**2])
Y = Matrix([rho, phi])
X.jacobian(Y)

# <codecell>

p=Function('p')(W*X)
p

# <codecell>

from sympy import Inverse, MatrixSymbol
x = MatrixSymbol('x',3,3)
M = Matrix([[3, -2, 4, -2], [5, 3, -3, -2], [5, -2, 2, -2], [5, -2, -3, 3]])
E,D=M.diagonalize()
E*D**-1/2*Transpose(E)*M*adjoint(E*D**-1/2*Transpose(E)*M)
D**-1/2*conjugate(E)*M*adjoint()
A = Symbol('A', hermitian=True)
B = Symbol('B', antihermitian=True)
adjoint(A)
x=E*D**-1/2*transpose(E)*E*D*

# <markdowncell>

# ## sympy 用于展示矩阵

# <codecell>

from sympy import Matrix
from sympy import *
from sympy import MatrixSymbol

# <codecell>

x = MatrixSymbol('x',3,3)
y= x.adjoint

# <markdowncell>

# 这里必须定义矩阵大小

# <codecell>

E*Transpose(E)
x = MatrixSymbol('x')
M = Matrix([[3, -2, 4, -2], [5, 3, -3, -2], [5, -2, 2, -2], [5, -2, -3, 3]])
E,D=M.diagonalize()
E*D**-1/2*Transpose(E)*M*adjoint(E*D**-1/2*Transpose(E)*M)
E*Adjoint(E)
x
type(x)
x*Adjoint(x)
x*Transpose(x)
X = MatrixSymbol('X', 3, 3)
X.adjoint()
X= Matrix([[3, -2, 4, -2], [5, 3, -3, -2], [5, -2, 2, -2], [5, -2, -3, 3]])
X*X.adjoint()
X.is_Identity()
X.is_Identity
A = Symbol('A', hermitian=True)
B = Symbol('B', antihermitian=True)
adjoint(A)
=
inv(A)
x = MatrixSymbol('x')
x = MatrixSymbol('x',3,3)
inv(x)
from sympy import *
(a1,a2,a3,b1,b2,b3) = symbols('a1 a2 a3 b1 b2 b3')
AB = Matrix([[a1*b1,a1*b2,a1*b3],[a2*b1,a2*b2,a2*b3],[a3*b1,a3*b2,a3*b3]])
print AB
print AB.det()
AB
ln(AB)
x
x.det()
AB.det()
log(AB)
logm(AB)
Matrix('x')

# <codecell>

Matrix('x')
AB
AB.det()
AB
Matrix([
[a1*b1, a1*b2, a1*b3],
[a2*b1, a2*b2, a2*b3],
[a3*b1, a3*b2, a3*b3]])
A=Matrix([[a1,a2],[b1,b2]])
from sympy.abc import *
a,b,c= symbols('a,b,c',real=True)
A=Matrix([[a,b],[b,c]])
X=log((A.det()))
pprint(X.diff(a))

# <markdowncell>

# ##可视化符号推倒

# <codecell>

from sympy.plotting import plot

# <codecell>

K=1024
pi =3.14
R =sin(pi*tau)/sin(pi*tau/K)
plot(R,(tau,-2,3))

# <codecell>

n = symbols('n',real = True)
N = symbols('N',real = True)
print summation(1,(n,1,N))
print summation(h(n),(n,1,N))

# <codecell>

##求极限
y1 = limit(sin(x)/x,x,0) 
y2= summation(x**n/factorial(n),(n,0,oo))
y2

# <codecell>


