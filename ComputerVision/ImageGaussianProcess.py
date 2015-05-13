# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# [gaussian process 分类图像]( http://www.pyimagesearch.com/2014/09/22/getting-started-deep-learning-python/)

# <codecell>

from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
from sklearn import datasets
from sklearn import gaussian_process
import numpy as np
import cv2
from sklearn.datasets.mldata import fetch_mldata
import pickle

# <codecell>

mnist = fetch_mldata('MNIST original')

# <markdowncell>

# 首次获取数据使用 fetch_mldata， 
# dataset = fetch_mldata('MNIST Original')
# 
# 下次就可以使用 load 文件了
# fid  = open('MnistData.pkl','wb')
# pickle.dump(dataset,fid)

# <markdowncell>

# 串行化数据导入

# <codecell>

dataset = fetch_mldata('iris')

# <codecell>

(trainX,testX,trainY,testY) = train_test_split(dataset.data/255.0,dataset.target.astype('int0'),test_size =0.33)
print trainX.shape, testX.shape

# <codecell>

gp = gaussian_process.GaussianProcess(theta0=1e-2,thetaL=1e-4,thetaU=1e-1)

# <codecell>

gp.fit(trainX[0:2],trainY[0:2])

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


