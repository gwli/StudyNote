# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # MOOC 数据分析

# <codecell>

import pandas as pd 
import matplotlib.pyplot as plt

# <codecell>

MoocData = pd.read_csv('HMXPC13_DI_v2_5-14-14.csv')

# <headingcell level=3>

# 不同课程的通过率

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['course_id'])
data = Certified.mean()
data.plot(kind='bar')

# <codecell>

data

# <markdowncell>

# 这些课是否都一样，是不一样的，这说明有些课比较松，还是讲的比较好，大家都通过了

# <markdowncell>

# ## 首先分析各个因子和是否拿到证书的关系

# <markdowncell>

# ### 活动次数和最后拿到证书的关系

# <markdowncell>

# 按照是否学够一半和认证的关系

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['explored'])
Certified.mean()

# <markdowncell>

# 没有读够一半的，一般没有获得证书。通过这个数据可以看出来，学习过半的通过的可能行显著提高

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['viewed'])
Certified.mean()

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['final_cc_cname_DI'])
data = Certified.mean()
data.plot(kind='bar')

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['LoE_DI'])
data = Certified.mean()
data.plot(kind='bar')

# <markdowncell>

# 说明这个课程具有通用性，所有人都能学习的差不多

# <headingcell level=3>

# 学习者出生年月日

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['YoB'])
data = Certified.mean()
data.plot(kind='bar')

# <codecell>

data[data==data.max()]

# <markdowncell>

# 这里的1939是什么意思，为什么会出现这个奇怪的数据?

# <headingcell level=3>

# 性别和通过与否的关系

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['gender'])
data = Certified.mean()
data.plot(kind='bar')

# <markdowncell>

# 说明性别基本没有关系

# <headingcell level=3>

# 在论坛的活跃程度和通过可能性的关系

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['ndays_act'])
data = Certified.mean()
data.plot(kind='bar')

# <markdowncell>

# 说明网上学习时间越长，通过的可能性越长

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['nforum_posts'])
data = Certified.mean()
data.plot(kind='bar')

# <markdowncell>

# 这个说明在论坛里面讨论的越多，通过的可能性越大

# <headingcell level=3>

# 感兴趣的章节数和通过的关系

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['nchapters'])
data = Certified.mean()
data.plot(kind='bar')

# <markdowncell>

# 说明感兴趣的越多，通过可能越大

# <headingcell level=3>

# 看视频和通过的关系

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['nplay_video'])
data = Certified.mean()
data.plot(kind='bar')

# <codecell>

Certified =MoocData['certified'].groupby(MoocData['incomplete_flag'])
data = Certified.mean()
data.plot(kind='bar')

# <headingcell level=3>

# 学习持续持续长短和通过率的关系

# <codecell>

Data =MoocData[MoocData['time_spend']>=0]
Certified =Data['certified'].groupby(Data['time_spend'])
data = Certified.mean()
data.plot(kind='bar')

# <markdowncell>

# 这个能说明什么？有些奇怪，需要解释

# <codecell>


