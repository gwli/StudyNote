# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import collections

# <headingcell level=3>

# 列出所有的可以选择的sheet 

# <codecell>

BaoKaoDate = pd.ExcelFile("TouDang.xlsx")
SheetNum =len(BaoKaoDate.sheet_names)-1
for i in xrange(SheetNum):
    print BaoKaoDate.sheet_names[i]

# <codecell>

for i in xrange(SheetNum):
    data = BaoKaoDate.parse(BaoKaoDate.sheet_names[1])

# <headingcell level=3>

# 一本分数线控制

# <codecell>

result = []
for i in xrange(SheetNum/4):
    dataL=[]
    dataR =[]
    dataLR=[]
    dataL = BaoKaoDate.parse(BaoKaoDate.sheet_names[i])
    dataR = BaoKaoDate.parse(BaoKaoDate.sheet_names[i+SheetNum/2])
    #print BaoKaoDate.sheet_names[i]
    dataLR = dataL.append(dataR)
    dataLR.drop_duplicates(cols=u'院校名称', take_last=True, inplace=True)
    data= pd.DataFrame(dataLR,columns=[u'院校代号',u'院校名称',u'计划人数',u'实际投档人数', u'超出'])
    data.interpolate(method='bfill')
    data2=data[data[u'超出']<=20]
    data3=data2[data2[u'计划人数']>20]
    data4=data3[data3[u'实际投档人数']-data3[u'计划人数']<data3[u'计划人数']*0.1]
    data5= pd.DataFrame(data4,columns=[u'院校代号'])
    result.append(data5)

# <codecell>

result = pd.concat(result, axis=0) 
for x,y in collections.Counter(result[u'院校代号']).items():
    if y>3:
        print int(x)
    

# <codecell>


# <codecell>

result= pd.DataFrame(data,columns=[u'院校代号'])
print [x for x,y in collections.Counter(result[u'院校代号']).items() if y>1]

# <codecell>

df

# <codecell>

result.code('utf8')

# <codecell>

lis = ['hoo', 'boo', 'hoo']
lis.append('boo')
lis

# <codecell>

print [x for x, y in collections.Counter(lis).items() if y > 1]

# <codecell>

现在要查找重复的。

# <codecell>

import collections
print [x for x,y in collections.Counter(results).items() if y>1]

# <markdowncell>

# parse 是什么意思？

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


