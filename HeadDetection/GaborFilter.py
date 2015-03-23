# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cv2
import numpy as np
from glob import glob
from IPython.display import Image

# <markdowncell>


# <codecell>

def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0,np.pi, np.pi/16):
        kern  = cv2.getGaborKernel((ksize,ksize), 4.0, theta, 10.00, 0.5, 0, ktype = cv2.CV_32F)
        kern /= 1.5* kern.sum()
        filters.append(kern)
    return filters

# <markdowncell>

# 这里的getGaborKernel 实现什么功能？

# <codecell>

def porcess(img,filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum,fimg,accum)
    return accum

# <markdowncell>

# 这里选出最大的gabor filter值，干什么？

# <codecell>

if __name__ == '__main__':
    images = glob("image_data/image_0001.jpg")
    img = cv2.imread(images[0])
    filters = build_filters()
    filters.shape
    res = porcess(img, filters[0])
    cv2.imwrite('result.jpg',res)
    

# <markdowncell>

# 现在这个gabor filter是什么？还需要再写写

# <codecell>

Image('image_data/image_0001.jpg',height=200,width=200)

# <codecell>

Image(filename='result.jpg') 

# <markdowncell>

# 这里得到的是什么信息？这么奇怪

# <markdowncell>

# 详细参考：http://visioncompute.readthedocs.org/en/latest/ComputerVision/GaborFilter.html?highlight=gabor

