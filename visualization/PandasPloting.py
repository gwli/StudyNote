# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

#  pandas 进行数据可视化

# <codecell>

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# <codecell>

import inspect

# <codecell>

matplotlib.__version__
pd.options.display.mpl_style ='default'

# <markdowncell>

#   pd.options.display.mpl_style ='default' 用来产生更多好看的图

# <headingcell level=2>

# Series： 一维数组操作

# <headingcell level=3>

# 数据结构包含 键/值 ,格式如下： Series (data,index = )

# <codecell>

N=10
ts = pd.Series(randn(N,),index = pd.date_range('1/1/2000',periods = N))
pd.Series()
ts

# <markdowncell>

#  pd.Series()产生序列， 这里的index是包含特定格式的index 1/1/2000。 1000个数据。
# 

# <markdowncell>

# index = pd.date_range('1/1/2000',periods = 1000)是list中函数意思是从2000-1-1日开始连续产生1000个日期

# <headingcell level=3>

# 支持滤波, array 操作, 查找

# <codecell>

ts[ts>0]
np.exp(ts)
'1/1/2000' in ts.index
'0' in ts

# <headingcell level=3>

# 可以从原来的序列中继续抽取序列

# <codecell>

ts1 = pd.Series(ts,index=pd.date_range('1/8/2000',periods=N) )
ts1.isnull()

# <markdowncell>

# 对于没有 index的数据显示不存在,并可继续判断不存在,从而可以判断那些数据丢失与否

# <headingcell level=3>

# 支持数据交并补操作

# <codecell>

tss =ts+ts1
tss[tss.isnull()]

# <headingcell level=3>

# 支持加name

# <codecell>

ts.index.name ="datas"
ts.name ="results"

# <codecell>

ts = ts.cumsum()
ts.plot()

# <markdowncell>

# ts.cumsum 按照要求的轴返回累积值，等会试下其他的,
# 好像是没有用的。

# <markdowncell>

# 但是这个数据只能是一维的，无法是二维的.

# <codecell>

ts.plot()

# <headingcell level=2>

# DataFrame： 多数数组操作

# <codecell>

data = {'state': ['Ohio', 'Ohio', 'Ohio', 'Nevada', 'Nevada'],
'year': [2000, 2001, 2002, 2001, 2002],
'pop': [1.5, 1.7, 3.6, 2.4, 2.9]}
frame =pd.DataFrame(data)

# <headingcell level=3>

# 按照一定的顺序调整数据显示形式

# <codecell>

frame1 =pd.DataFrame(data,columns=['year','pop'])

# <headingcell level=3>

# 如果选取的列不存在，则输出NaN:
#     

# <codecell>

frame2 = pd.DataFrame(data,columns=['year','pop','debt','state'])
frame2

# <codecell>

pd.DataFrame(data).columns

# <headingcell level=3>

# 显示某一列的数值

# <codecell>

frame_year =pd.DataFrame(data).year
frame_year

# <headingcell level=3>

# 查看某一个序号（index）下的data

# <codecell>

frame.ix[3]

# <rawcell>

# 这里为什么必须要 .ix 那？  这里的index 其实是虚的.

# <markdowncell>

# 对 “debt”赋值

# <codecell>

frame2['debt'] =9
frame2

# <codecell>

frame2['debt'] = np.arange(1,10,2)
frame2

# <headingcell level=3>

# 合并数据: 对于两本版本中不一致的数据，按照序号进行合并

# <codecell>

val = pd.Series([-1.2, -1.5, -1.7],index=[0,2,3])
frame2['debt'] = val
frame2

# <markdowncell>

# 对于数据中没有的一列，直接新加一行 

# <codecell>

frame2['eastern']= frame2.state =='Ohio'
frame2

# <codecell>

del frame2['eastern']
frame2

# <codecell>

pop = {'Nevada': {2001: 2.4, 2002: 2.9}, 'Ohio': {2000: 1.5, 2001: 1.7, 2002: 3.6}}

# <codecell>

frame3 = pd.DataFrame(pop)
frame3.T

# <headingcell level=3>

# 选取数据

# <codecell>

DataFrame(pop, index=[2001, 2002, 2003])

# <codecell>

frame3.index.name='year'
frame3.columns.name='state'
frame3.values
frame3.index
frame3.columns

# <codecell>

df = pd.DataFrame(randn(1000,4),index=ts.index,columns=list('ABCD'))

# <codecell>

df = df.cumsum()
df.plot()

# <codecell>

plt.figure()
df.plot()

# <markdowncell>

# 上面的两个有什么区别

# <codecell>

df3 = pd.DataFrame(randn(1000,2),columns=['B','C']).cumsum()

# <markdowncell>

# 什么意思 colunns 是什么意思

# <codecell>

df3['A'] = pd.Series(list(range(len(df))))

# <markdowncell>

# list 是分开排列的意思

# <codecell>

df3.plot(x ='A',y ='B',legend='gj')

# <markdowncell>

# 这里 legnd 没有用，并且y label也没有用

# <codecell>

plt.figure()
df.ix[5].plot(kind='bar')

# <markdowncell>

# plt.ix[5]是什么意思

# <codecell>

df4 = pd.DataFrame(randn(10,3),columns=list('ABC'))
df4.plot(kind ='bar')

# <markdowncell>

# 这里为什么要使columns ？

# <codecell>

df4.plot(kind = 'barh')

# <markdowncell>

# 这个表示横线

# <codecell>

a = randn(1000)
a.shape

# <codecell>

df4.plot(kind = 'hist',bins = 10)

# <markdowncell>

# 使用横着的图

# <codecell>

df4.plot(kind = 'hist',orientation ='horizontal', bins = 10)

# <codecell>

plt.figure()
df.diff().hist()

# <markdowncell>

# 这个表示什么意思？

# <codecell>

df['A'].diff().hist()

# <markdowncell>

# 'A'表示图像A吗？

# <codecell>

df = pd.DataFrame(rand(10,4),columns=list('ABCD'))
df.plot(kind ='area')

# <markdowncell>

# ## scatter plot

# <codecell>

df = pd.DataFrame(rand(50,4),columns=list('abcd'))
df.plot(kind ='scatter',x ='a',y='c')

# <codecell>

ax = df.plot(kind ='scatter',x ='a',y ='b',color = 'DarkBlue',label ='Group 1')
ax = df.plot(kind ='scatter',x ='c',y ='d',color = 'DarkGreen',label ='Group 2')

# <markdowncell>

# columuns 和 labe 有什么区别

# <codecell>

df.plot(kind ='scatter',x='a',y='b',s=120)

# <codecell>

df.plot(kind='scatter', x='a', y='b', c='c', s=120);

# <markdowncell>

# 这里的'c'是什么意思，加入坐标

# <markdowncell>

# ## Hexagonal Bin Plot

# <markdowncell>

# ## Pie Plot

# <codecell>

series = pd.Series(2*rand(4),index=list('abbc'),name ='series')
series.plot(kind ='pie',autopct = '%.2f',figsize=(8,8))

# <markdowncell>

# 这个df 为什么是矩阵，而现实的又是另外一个样子？

# <codecell>

from pandas.tools.plotting import scatter_matrix
df = pd.DataFrame(randn(100,4),columns= list ('abcd'))
scatter_matrix(df,alpha=0.4,figsize=(10,10),diagonal = 'kde')

# <markdowncell>

# ## Density Plot

# <codecell>

ser = pd.Series(rand(100))
ser.plot(kind ='kde')

# <markdowncell>

# ## Andrews Curves

# <markdowncell>

# 可以画多个图线，使用傅里叶级数，什么意思

# <codecell>

from pandas import  read_csv
from pandas.tools.plotting import andrews_curves
#data = read_csv('data/iris.data')

# <markdowncell>

# ## Lag Plot

# <markdowncell>

# 检测数据是否是随机数据？

# <codecell>

from pandas.tools.plotting import lag_plot
plt.figure()
data = pd.Series(0.1*uniform(1000)+0.9*np.sin(np.linspace(-99*np.pi,99*np.pi,num=1000)))
lag_plot(data)

# <rawcell>

# 测试一个均匀数据

# <codecell>

import numpy as np 
from pandas import DataFrame
import matplotlib.pyplot as plt 

Index= ['aaa', 'bbb', 'ccc', 'ddd', 'eee']
Cols = ['A', 'B', 'C', 'D']
df = DataFrame(abs(np.random.randn(5, 4)), index=Index, columns=Cols)

plt.pcolor(df)
plt.yticks(np.arange(0.5, len(df.index), 1), df.index)
plt.xticks(np.arange(0.5, len(df.columns), 1), df.columns)
plt.show()

# <codecell>

p4d = pd.Panel4D(randn(2, 2, 5, 4),
   ....:       labels=['Label1','Label2'],
   ....:       items=['Item1', 'Item2'],
   ....:       major_axis=pd.date_range('1/1/2000', periods=5),
   ....:       minor_axis=['A', 'B', 'C', 'D'])
p4d

# <markdowncell>

# 这个是什么意思

# <codecell>


