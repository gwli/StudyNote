# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import pandas as pd
import collections

# <headingcell level=3>

# 设定考生考分阀值

# <codecell>

def ChooseSchool(RangeStart,RangeEnd,ChaoChuThreshold1,ChaoChuThreshold2,JHRS,Ratio):
    """
    设定超出分数线分数
    设定投档人数下限
    设定实际投档的人数不能超过计划的百分比
    RangeStart 表示从那个数据开始
    RangeEend 从哪个数据结束
    ChaoChuThreshold1 最低超出分数线多少分，跟一本还是二本有关系
    ChaoChuThreshold2 最高超出分数线多少分，跟一本还是二本有关系
    JHRS 计划报考人数
    Ratio 不超过计划的比例
    """
    result = []
    allData=[]
    for i in xrange(RangeStart,RangeEnd):
        data=[]
        data = BaoKaoDate.parse(BaoKaoDate.sheet_names[i])
        data= pd.DataFrame(data,columns=[u'院校代号',u'院校名称',u'计划人数',u'实际投档人数', u'超出'])
        allData.append(data)
        data.interpolate(method='bfill')
        data1=data[data[u'超出']>ChaoChuThreshold1]
        data2=data1[data1[u'超出']<=ChaoChuThreshold2]
        data3=data2[data2[u'计划人数']>JHRS]
        data4=data3[data3[u'实际投档人数']-data3[u'计划人数']<data3[u'计划人数']*Ratio]
        data5= pd.DataFrame(data4,columns=[u'院校代号'])
        result.append(data5)
    result = pd.concat(result, axis=0) 
    allData = pd.concat(allData,axis=0)
    for x,y in collections.Counter(result[u'院校代号']).items():
        if y>2:
            print int(x)
            print allData[allData[u'院校代号']==x]
            allData.to_csv('results.csv',encoding='utf-8')

# <headingcell level=3>

# 列出所有的可以选择的sheet 

# <codecell>

BaoKaoDate = pd.ExcelFile("TouDangWen.xlsx")
SheetNum =len(BaoKaoDate.sheet_names)-1
for i in xrange(SheetNum):
    print BaoKaoDate.sheet_names[i]

# <headingcell level=3>

# 一本院校分析

# <codecell>

#ChooseSchool(0,SheetNum/2,10,50,0,10)

# <headingcell level=3>

# 二本院校分析

# <codecell>

ChooseSchool(SheetNum/2,SheetNum,0,25,0,10)

