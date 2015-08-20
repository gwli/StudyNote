# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import numpy as np
from glob import glob
import pandas as  pd
import os
import cv2

# <codecell>

BeforePoressing = glob('zp/*.jpg')

# <codecell>

N= size(BeforePoressing)
N

# <codecell>

BiDuiBiao = pd.read_csv('BiDuiBiao.csv')

# <codecell>

BiDuiBiao['KaoShengHao1'][0]

# <codecell>

filename[3:18]

# <codecell>

count =0
for elem in BeforePoressing:
    (filename,ext) = os.path.splitext(elem)
    image = cv2.imread(elem)
    KaoShengHao =filename[3:18]
    KaoShengHaoBiao = BiDuiBiao['KaoShengHao1']
    value = KaoShengHaoBiao[int(KaoShengHao)==KaoShengHaoBiao].values.tolist()
    if value!=[]:
        cv2.imwrite(str(value[0])+'.jpg',image)  
        count+=1

# <codecell>

str(value[0])

# <codecell>

count

# <codecell>

ls 15411

# <codecell>

ls 150301230093.jpg

# <codecell>


