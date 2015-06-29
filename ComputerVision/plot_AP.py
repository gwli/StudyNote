# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

cd face-eval/

# <codecell>

#!/usr/bin/env python

#import matplotlib
# matplotlib.use('Agg')
import sys
from database import *
import VOCpr
import os
import numpy as np
from loadData import loadDetections
from getColorLabel import *
from VOCpr import evaluate_optim, filterdet

# <codecell>

minw =30
minh = 30
nit =5
#detfile = "detections/results.csv"

minpix = int(np.sqrt(0.5 * minw * minh))
baseFolder = "detections/AFW"
tsImages = getRecord(AFW(minw=minw, minh=minh), 10000)
pylab.figure(figsize=(10, 7))
res = []
for fn in glob.glob(os.path.join(baseFolder, "*")):
    print fn
    ovr = 0.5
    is_point = False
    ff = os.path.basename(fn).split(".")
    if ff[0] == "Face++" or ff[0] == "Picasa" or ff[0] == "Face":
        is_point = True
    if ff[0] == "Picasa":  # and ff[1]=="cvs":
        # special case for picasa
        # we evaluated manually, selecting overlap threshold of 0.1 gives
        # in this case the correct result
        ovr = 0.1
    dets = loadDetections(fn)
    dets = filterdet(dets, minpix)
    color, label = getColorLabel(fn)
    r = evaluate_optim(
        tsImages, dets, label, color, point=is_point, iter=nit, ovr=ovr)
    res.append(r)

# <codecell>


# <codecell>



# current plot
if detfile != "":
    dets = loadDetections(detfile)
    dets = filterdet(dets, minpix)
    r = evaluate_optim(tsImages, dets, detfile, 'green', iter=nit)
    res.append(r)


res.sort(key=lambda tup: tup[0], reverse=True)
ii = []
ll = []

for this_idx, i in enumerate(res):
    idx = i[3]
    plot_id = i[1]
    label = i[2]
    ii.append(plot_id)
    ll.append(label)
    #print (this_idx)
    if pylab.getp(plot_id, 'zorder') < 40:
        pylab.setp(pylab.findobj(plot_id), zorder=len(res) - this_idx)

pylab.legend(ii, ll, loc='lower left')
pylab.xlabel("Recall")
pylab.ylabel("Precision")
pylab.grid()
pylab.gca().set_xlim((0, 1))
pylab.gca().set_ylim((0, 1))
savename = "%s_final.pdf" % args.dataset
pylab.savefig(savename)
os.system("pdfcrop %s" % (savename))
pylab.show()
pylab.draw()

# <codecell>


