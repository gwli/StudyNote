# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import glob,os,cv2
files = glob.glob("bikes_and_persons/*.bmp")


# <codecell>

k=1
for elem in files:
   print elem
   (filename,ext) = os.path.splitext(elem)
   im = cv2.imread(elem)
   cv2.imwrite(filename+".jpg",im)
   k=k+1

# <codecell>

del  *.bmp

# <codecell>


