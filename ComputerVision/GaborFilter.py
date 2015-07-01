# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# Gabor filter 

# <markdowncell>

# $   H\left( {u,v} \right) = 2\pi {\delta _x}{\delta _y}\left[ {{e^{ - 2{\pi ^2}\left[ {{{\left( {\mu  - {\mu _0}} \right)}^2}\delta _x^2 + {{\left( {v - {v_0}} \right)}^2}\delta _y^2} \right]}}} \right]
# $

# <codecell>

def build_filters():
    filters = []
    ksize = 31
    #for theta in np.arange(0,np.pi, np.pi/16):
    for theta in np.arange(0, np.pi, np.pi/2):
        kern  = cv2.getGaborKernel((ksize,ksize), 4.0, theta, 10.00, 0.5, 0, ktype = cv2.CV_32F)
        kern /= 1.5* kern.sum()
        filters.append(kern)
        pl
    return filters

# <markdowncell>

# 这里的getGaborKernel 实现什么功能？
# 返回 Gabor滤波系数

# <codecell>

import cv2
import numpy as np
from glob import glob
from IPython.display import Image
import matplotlib.pyplot as plt

# <codecell>

def porcess(img,filters):
    accum = np.zeros_like(img)
    plt.figure(figsize=(10,10))
    i = 1 
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        plt.subplot(4,4,i)
        cv2.imwrite('result.jpg',res)
        i+=1
        Image(filename='result.jpg')
        np.maximum(accum,fimg,accum)
    return accum

# <markdowncell>

# 这里选出最大的gabor filter值，干什么？

# <codecell>

if __name__ == '__main__':
    images = glob("image_data/image_0001.jpg")
    img = cv2.imread(images[0])
    filters = build_filters()
    res = porcess(img, filters)
    cv2.imwrite('result.jpg',res)

# <markdowncell>

# 现在这个gabor filter是什么？还需要再写写

# <codecell>

Image('image_data/image_0001.jpg',height=200,width=200)

# <codecell>

Image(filename='result.jpg') 

# <markdowncell>

# gabor filter 获得的是方向信息。这里是所有方向信息的混合。
# 详细参考：http://visioncompute.readthedocs.org/en/latest/ComputerVision/GaborFilter.html?highlight=gabor

# <markdowncell>

# gabor filter 获得的是方向信息，为什么使用gabor filter 作为 神经网络的第一层？

# <markdowncell>

# 后面是不是自己写gabor filter更细节的东西，来验证写什么？  这里为什么要多尺度那？

# <codecell>


