# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd

# <headingcell level=3>

# 列出所有的可以选择的sheet 

# <codecell>

BaoKaoDate = pd.ExcelFile("TouDang.xlsx")
SheetNum =len(BaoKaoDate.sheet_names)-1
for i in xrange(SheetNum):
    print BaoKaoDate.sheet_names[i]

# <codecell>

results=[]
for i in xrange(SheetNum):
    data = BaoKaoDate.parse(BaoKaoDate.sheet_names[1])

# <codecell>

results=[]
for i in xrange(SheetNum/2):
    dataL=[]
    dataR =[]
    dataLR=[]
    dataL = BaoKaoDate.parse(BaoKaoDate.sheet_names[i])
    dataR = BaoKaoDate.parse(BaoKaoDate.sheet_names[i+SheetNum/2])
    dataLR = dataL.append(dataR)
    dataLR.drop_duplicates(cols=u'院校名称', take_last=True, inplace=True)
    data= pd.DataFrame(dataLR,columns=[u'院校代号',u'院校名称',u'计划人数',u'实际投档人数', u'最低投档分'])
    #print dataL.shape,dataR.shape,dataLR.shape
    data.interpolate(method='bfill')
    results.append(data)  

# <codecell>

print BaoKaoDate.sheet_names[i],data

# <codecell>

    data1=data[data[u'最低投档分']<=567]
    data1=data[data[u'最低投档分']<=567]
    data3=data2[data2[u'计划人数']>20]
    data4=data3[data3[u'实际投档人数']-data3[u'计划人数']<data3[u'计划人数']*0.1]
    results.append(data4)  

# <codecell>

这里分数线还没有细分到

# <headingcell level=3>

# 设定搜索一本二本范围

# <codecell>

results=[]
for i in xrange(SheetNum/2):
    data = BaoKaoDate.parse(BaoKaoDate.sheet_names[i])
    data1 =data.drop([1],axis=0)
    data2=data1[data1['最低投档分']<=567]
    data3=data2[data2[u'计划']>20]
    data4=data3[data3['Unnamed: 3']-data3['Unnamed: 2']<data3['Unnamed: 2']*0.1]
    results.append(data4)  
data4.shape
#print results
#pd.DataFrame(np.asarray(results))

# <headingcell level=3>

# 设定阀值，删除数据（可选）

# <codecell>

data.columns

# <codecell>

data1 =data.drop([1],axis=0)

# <headingcell level=3>

# 设定考生考分阀值

# <codecell>

data2.shape

# <codecell>

data3.shape

# <headingcell level=3>

# 投档的人数不能超过技术的20%

# <codecell>

data4.shape

# <codecell>

data4

# <codecell>


