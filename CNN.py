# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

from theano.tensor.nnet import conv
rng = numpy.random.RandomState(23455)
import theano.tensor as T
import theano
import numpy as np

# <codecell>

input = T.tensor4(name='input')
# 初始权值
w_shape = (2,3,9,9)
w_bound = np.sqrt(3*9*9)
b_shape = (2,)  
W = theano.shared(np.asarray(rng.uniform(low = -1.0/w_bound,high = 1.0/w_bound,size = w_shape),dtype= input.dtype),name= 'W')
b = theano.shared(np.asarray(rng.uniform(low=-.5, high=.5, size=b_shape),dtype=input.dtype), name ='b')
conv_out = conv.conv2d(input, W)
output = T.nnet.sigmoid(conv_out + b.dimshuffle('x', 0, 'x', 'x'))
f = theano.function([input], output)

# <markdowncell>

# b.dimshuffle('x', 0, 'x', 'x') 是什么？ 中间有0
# 这里是直接卷积那？还是有优化？

# <codecell>

import pylab
from PIL import Image
img = Image.open(open('image_data/image_0003.jpg'))
[width,height]=img.size

# <codecell>

img = np.asarray(img, dtype='float64') /256
img_ = img.swapaxes(0, 2).swapaxes(1, 2).reshape(1, 3, width,height)
filtered_img = f(img_)
pylab.subplot(1, 3, 1); pylab.axis('off'); pylab.imshow(img)
pylab.gray()
pylab.subplot(1, 3, 2); pylab.axis('off'); pylab.imshow(filtered_img[0, 0, :, :])
pylab.subplot(1, 3, 3); pylab.axis('off'); pylab.imshow(filtered_img[0, 1, :, :])
pylab.show()

# <markdowncell>

# 现在这个图是不对的，怎样进行tiaoshi？

