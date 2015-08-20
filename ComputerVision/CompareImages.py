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

a=10
b=120
c =10
d=80
NN = (b-a)*(d-c)
for elem in files[1:10]:
    (filename,ext) = os.path.splitext(elem)
    BeforeProcessingImage= cv2.imread(elem)
    BeforeProcessingImage.resize((126,126))
    KaoShengHao =filename[3:18]
    KaoShengHaoBiao = BiDuiBiao['KaoShengHao1']
    index = KaoShengHaoBiao[int(KaoShengHao)==KaoShengHaoBiao].index[0]
    if value!=[]:
        AfterProcessingImage= cv2.imread('AP/'+str(XueHao[0])+'.jpg')
        AfterProcessingImage.resize((126,126))
        error=(BeforeProcessingImage[a:b,c:d]-AfterProcessingImage[a:b,c:d])/NN
 

# <codecell>

KaoShengHaoBiao.index(index)

# <codecell>

KaoShengHao

# <codecell>

XueHao

# <codecell>


