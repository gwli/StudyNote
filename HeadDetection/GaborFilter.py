# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cv2
import numpy as np
from glob import glob

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
    images = glob("image_data/*.jpg")
    img = cv2.imread(images[0])
    filters = build_filters()
    res1 = porcess(img, filters)
    cv2.imshow('result',img)
    cv2.waitKey()
    cv2.destroyAllWindows()

# <markdowncell>

# 现在这个gabor filter是什么？还需要再写写

