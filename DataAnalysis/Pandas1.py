# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import  pandas as pd

# <codecell>

pd.read_csv('ex2.csv',indexs=['a','b','c'])

# <codecell>

data = pd.read_csv('ex2.csv',names=['a','b','c','d'])

# <markdowncell>

# 如果names不完全，就显示不完全

# <codecell>

pd.read_excel('ex2.csv',index_col ='message')

# <codecell>

chunker = pd.read_csv('ex2.csv',chunksize=2)

# <codecell>

chunker

# <codecell>

tot = pd.Series([])

# <codecell>

for piece in chunker:
    #tot = tot.add(piece.values value_counts(),fill_value=0)
    print piece.values
#tot = tot.order(ascending=True)

# <headingcell level=3>

# 写数据

# <codecell>

data.to_csv('out.csv')

# <codecell>

ls out.csv

# <codecell>

pd.read_csv('out.csv')

# <markdowncell>

# 怎样控制第一列？

# <codecell>


# <codecell>

data.to_csv(sys.stdout,index=True,header=True)

# <codecell>

data.index=[2,24,4]
data

# <markdowncell>

# 这里可以更改的data.index

# <codecell>

data = pd.read_csv('ex2.csv',names=['a','b','c','d','e'])

# <markdowncell>

# 如果没有数据，自动把第一个数据当做训练。

# <codecell>


