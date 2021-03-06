# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from __future__ import division
import cv2
import numpy as np
import math
import matplotlib.pyplot as plt
import os
import argparse
from IPython.display import  Image

# <codecell>

imagePath ='HeadImages/2841_287304_484811.jpg'

# <codecell>

img = cv2.imread(imagePath)
img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
img = cv2.GaussianBlur(img,(3,3),0) 

# <codecell>

Image(imagePath)

# <codecell>

cropImg = img
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
erodedImg = cv2.erode(cropImg,kernel) 
height,width = cropImg.shape
tmp =[]
size = 15
for i in range(1,height-size):
    for jj in range(1,width-size):
        if (1.5*erodedImg[i:i+size,jj]<np.mean(erodedImg)).all() and (1.5*erodedImg[i,jj:jj+size]<np.mean(erodedImg)).all():
            tmp.append([i,jj])
            cv2.circle(img,(jj+np.int(size/2),i+np.int(size/2)),np.int(size/2),(255,0,0))
cv2.imwrite('detectedPoints.jpg',img)
tmp = np.array(tmp)

# <codecell>

Image('detectedPoints.jpg')

# <codecell>

os.remove('detectedPoints.jpg')

# <codecell>


