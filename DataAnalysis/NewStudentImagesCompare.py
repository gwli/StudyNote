# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from glob import glob
import pandas as  pd
import os
import cv2

# <codecell>

os.getcwd()

# <codecell>

BeforePoressing = glob('BeforeProcessing/*.jpg')

# <codecell>

BeforePoressing[0][17:32]

# <codecell>

N= size(BeforePoressing)

# <codecell>

BeforePoressing = glob('BeforeProcessing/*.jpg')

# <codecell>

BiDuiBiao = pd.read_csv('BiDuiBiao.csv')

# <codecell>

BiDuiBiao

# <codecell>

BiDuiBiao['KaoShengHao1'][0]

# <codecell>

int(KaoShengHao)

# <codecell>

for elem in BeforePoressing:
    (filename,ext) = os.path.splitext(elem)
    image = cv2.imread(elem)
    KaoShengHao =filename[17:35]
    KaoShengHaoBiao = BiDuiBiao['KaoShengHao1']
    index = KaoShengHaoBiao[KaoShengHaoBiao==int(KaoShengHao)].index[0]
    cv2.imwrite(str(BiDuiBiao['XueHao1'][index])+'.jpg',image)        

# <codecell>

BiDuiBiao['XueHao1'][0]

# <codecell>

ls 150.jpg  

# <codecell>

ls 150301230093.jpg

# <codecell>


