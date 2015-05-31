# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# [随机森林](http://blog.yhathq.com/posts/random-forests-in-python.html)

# <codecell>

from sklearn.datasets import  load_iris
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np

# <codecell>

iris = load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)
df['is_train']= np.random.uniform(0,1,len(df))<=0.75
df['species']=pd.Categorical.from_codes(iris.target,iris.target_names)
df.head()

# <codecell>

train,test = df[df['is_train']==True],df[df['is_train']==False]

# <codecell>

features = df.columns[:4]

# <codecell>

clf = RandomForestClassifier(n_jobs=2)

# <codecell>

y,_=pd.factorize(train['species'])

# <codecell>

clf.fit(train[features],y)
preds = iris.target_names[clf.predict(test[features])]
pd.crosstab(test['species'],preds,rownames=['actual'],colnames=['preds'])

# <codecell>


