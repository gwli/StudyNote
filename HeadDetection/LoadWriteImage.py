# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cv2
from glob import glob

# <codecell>

#!ls image_data
images = glob("image_data/*.jpg")
images[0]

# <codecell>

img = cv2.imread(images[0])
img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
Nimg = cv2.resize(img,(8,8))

# <codecell>

cv2.imwrite('a.jpg',Nimg)

# <markdowncell>

# cv2.imshow('ImageWindow',Nimg)
# 
# 这个无法执行，总是重启机器，但是这样就无法看到执行结果。

