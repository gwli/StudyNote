# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd

# <codecell>

for i in xrange(len(BaoKaoDate.sheet_names)):
    print BaoKaoDate.sheet_names[i]

# <codecell>

BaoKaoDate = pd.ExcelFile("TouDang.xlsx")
data = BaoKaoDate.parse(BaoKaoDate.sheet_names[1])
print BaoKaoDate.sheet_names[1]
data.head()

# <markdowncell>

# 设定阀值，删除数据

# <codecell>

data.columns

# <codecell>

data1 =data.drop([1],axis=0)

# <headingcell level=3>

# 加入考生考分阀值

# <codecell>

data2=data1[data1['Unnamed: 4']<=567]
data2

# <codecell>

data3=data2[data2['Unnamed: 2']>20]
data3

# <headingcell level=3>

# 投档的人数不能超过技术的20%

# <codecell>

data4=data3[data3['Unnamed: 3']-data3['Unnamed: 2']<data3['Unnamed: 2']*0.1]
data4

# <codecell>

data4.shape

# <codecell>


