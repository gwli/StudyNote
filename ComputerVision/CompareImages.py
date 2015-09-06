# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import cv2, os
from glob import glob
import  pandas as pd

# <codecell>

BiDuiBiao = pd.read_csv('BiDuiBiao.csv')

# <codecell>

BeforeProcessing =glob('zp/*.jpg')
AfterProcessing = glob('AP/*.jpg')

# <codecell>

a = np.asarray([0,1,2,3,4])

# <codecell>

a.shape

# <codecell>

a=10
b=120
c =10
d=80
NN = (b-a)*(d-c)
for elem in BeforeProcessing:
    (path,filename) = os.path.split(elem)
    (KaoShengHao,ext) = os.path.splitext(filename)
    BeforeProcessingImage= cv2.imread(elem)
    BeforeProcessingImage.resize((126,126))
    KaoShengHaoBiao = BiDuiBiao['KaoShengHao1']
    index = KaoShengHaoBiao[KaoShengHaoBiao==int(KaoShengHao)]
    value = index.values
    if value!=[]:
        print 1

# <codecell>

if value!=[]:
    print 1

# <codecell>

  if value!=[]:
        AfterProcessingImage= cv2.imread('AP/'+str(XueHao[0])+'.jpg')
        AfterProcessingImage.resize((126,126))
        error=(BeforeProcessingImage[a:b,c:d]-AfterProcessingImage[a:b,c:d])/NN

# <codecell>

KaoShengHaoBiao

# <codecell>

int(KaoShengHao)

# <codecell>

int(KaoShengHao)

# <codecell>

KaoShengHaoBiao.index(index)

# <codecell>

KaoShengHao

# <codecell>

XueHao

# <codecell>


