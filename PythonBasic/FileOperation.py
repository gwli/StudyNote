# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

import tempfile
from sklearn import  datasets

# <codecell>

help(datasets.fetch_mldata)

# <codecell>

ls datasets

# <codecell>

import pickle

# <codecell>

d = { "abc" : [1, 2, 3], "qwerty" : [4,5,6] }

# <codecell>

afile = open(r'd.pkl','wb')

# <codecell>

help(pickle.dump)

# <codecell>

pickle.dump(d,afile)

# <codecell>

afile.close()

# <codecell>

fid= open('d.pkl','rb')

# <codecell>

new_file = pickle.load(fid)
new_file

# <codecell>


