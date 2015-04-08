# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# # 代码目的：
# ## 使用 mlxtend用于可视化，
# ## 使用支持向量机。

# <codecell>

from  mlxtend.evaluate import plot_decision_regions
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm  import SVC
from IPython.display import Image

# <codecell>

iris = datasets.load_iris()
x = iris.data[:,[0,2]]
y = iris.target
svm = SVC(C=0.5,kernel='linear')
svm.fit(x,y)

# <codecell>

plot_decision_regions(x,y,clf = svm, res=0.1,legend = 2)
plt.xlabel('sepal length [cm]')
plt.ylabel('petal length [cm]')
plt.title('SVM on Iris')
plt.show()

# <markdowncell>

# # 问题
# 
# ## 这里是x.shape[1] 必须等于2，目前没有处理高维数据的。

# <codecell>


