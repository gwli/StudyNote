# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# [图像分类](http://nbviewer.ipython.org/gist/hernamesbarbara/5768969)

# <codecell>

import pandas as pd
import numpy as np
import pylab as pl
import PIL
from PIL import  Image
import os
import base64
from StringIO import  StringIO

# <codecell>

from sklearn.decomposition import RandomizedPCA
from sklearn.neighbors import KNeighborsClassifier

# <codecell>

%matplotlib inline

# <codecell>

def img_to_matrix(filename, verbose=False):
    """
    takes a filename and turns it into a numpy array of RGB pixels
    """
    img = PIL.Image.open(filename)
    if verbose==True:
        print "changing size from %s to %s" % (str(img.size), str(STANDARD_SIZE))
    img = img.resize(STANDARD_SIZE)
    img = list(img.getdata())
    img = map(list, img)
    img = np.array(img)
    return img

# <markdowncell>

# 图像拉成列向量

# <codecell>

def flatten_image(img):
    s = img.shape[0]*img.shape[1]
    img_wide = img.reshape(1,s)
    return img_wide[0]

# <codecell>

img_dir = 
images =[img]

