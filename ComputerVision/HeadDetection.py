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
from IPython.display import Image

# <codecell>

Image('images.jpg')

# <codecell>

def inside(r, q):
    rx, ry, rw, rh = r
    qx, qy, qw, qh = q
    return rx > qx and ry > qy and rx + rw < qx + qw and ry + rh < qy + qh

# <codecell>

def draw_detections(img, rects, thickness = 1):
    for x, y, w, h in rects:
        # the HOG detector returns slightly larger rectangles than the real objects.
        # so we slightly shrink the rectangles to get a nicer output.
        pad_w, pad_h = int(0.15*w), int(0.05*h)
        cv2.rectangle(img, (x+pad_w, y+pad_h), (x+w-pad_w, y+h-pad_h), (0, 255, 0), thickness)
        cv2.imwrite('detected.jpg',img)

# <codecell>

img = cv2.imread('images.jpg')  
img = cv2.cvtColor(img, cv2.cv.CV_BGR2GRAY)
img = cv2.GaussianBlur(img,(3,3),0) 
#edges = cv2.Canny(img, 50, 150)  
#cv2.imwrite('edges.jpg',edges)
#lines = cv2.HoughLinesP(edges,1,np.pi/180, 100, minLineLength= 20, maxLineGap = 10) 

# <headingcell level=3>

# HOG特征

# <codecell>

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# <markdowncell>

# 这里为什么要使用SVM，是不是还可以使用其他的

# <codecell>

found,w = hog.detectMultiScale(img,winStride=(8,8),padding=(32,32),scale=1.05)
print found,w

# <codecell>

found_filtered =[]
enumerate(found)
for ri,r in enumerate(found):
    for qi,q in enumerate(found):
        if ri!=qi and inside(r,q):
            break
        else:
            found_filtered.append(r)

# <markdowncell>

# detectMultiScale是什么？

# <codecell>

found_filtered

# <codecell>

draw_detections(img,found)
draw_detections(img,found_filtered,3)

# <codecell>

print '%d (%d) found' % (len(found_filtered), len(found))
cv2.imwrite('results.jpg',img)

# <codecell>

Image('results.jpg',height=200,width=200)

# <codecell>


