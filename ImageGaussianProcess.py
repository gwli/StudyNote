# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# 使用[gaussian process 分类图像]( http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/)

# <codecell>

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn import gaussian_process
import numpy as np
import cv2

# <codecell>

print "[X] downloading data..."
dataset = datasets.fetch_mldata("MNIST Original")

# <codecell>

print dataset

# <codecell>

(trainX,testX,traniY,testY) = train_test_split(dataset.data/255.0,dataset.target.astype('int0'),test_size =0.33)

# <codecell>

gp = gaussian_process.GaussianProcess(theta0=1e-2,thetaL=1e-4,thetaU=1e-1)

# <codecell>

gp.fit(trainX,traniY)

# <markdowncell>

# 太大了无法执行？

# <codecell>

preds = gp.predict(testX)
print classification_report(testY,preds)

# <codecell>

for i in np.random.choice(np.arange(0,len(testY)),size=(10,)):
    pred = gp.predict(np.atleast_2d(testX[i]))
    img = (testX[i]*255).reshape((28,28)).astype('uint8')
    print "Actual digit is {0}, predicted {1}".format(testY[i], pred[0])
	cv2.imshow("Digit", image)
	cv2.waitKey(0) 

# <markdowncell>

# 现在基本可以了，应该在更大的程序集上跑

# <codecell>


