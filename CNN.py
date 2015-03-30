# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from theano.tensor.nnet import conv
rng = numpy.random.RandomState(23455)
import theano.tensor as T
import theano
input = T.tensor4(name='input')
#w_shp = (2, 3, 9, 9)
#w_bound = numpy.sqrt(3 * 9 * 9)
#W = theano.shared( numpy.asarray(
#rng.uniform(
#low=-1.0 / w_bound,
#high=1.0 / w_bound,
#size=w_shp),
#dtype=input.dtype), name ='W')
#b_shp = (2,)
#b = theano.shared(numpy.asarray(rng.uniform(low=-.5, high=.5, size=b_shp),dtype=input.dtype), name ='b')

# <codecell>

# 初始权值
w_shape = (2,3,9,9)
w_bound = np.sqrt(3*9*9)
b_shape = (2,)  
W = theano.shared(np.asarray(rng.uniform(low = -1.0/w_bound,high = 1.0/w_bound,size = w_shape),dtype= input.dtype),name= 'W')
b = theano.shared(numpy.asarray(rng.uniform(low=-.5, high=.5, size=b_shape),dtype=input.dtype), name ='b')

# <codecell>

conv_out = conv.conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
f = theano.function([input], output)

# <codecell>

import numpy
import pylab
from PIL import Image
img = Image.open(open('image_data/image_0001.jpg'))
[width,height]=img.size

# <codecell>

img = numpy.asarray(img, dtype='float64') /256
img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, width,height)
filtered_img = f(img_)
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray();
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()

# <codecell>


