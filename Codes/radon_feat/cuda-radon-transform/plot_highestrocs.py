import os,sys
import re
from glob import glob
import matplotlib.pyplot as plt
import cPickle
import numpy as np
from utils import *

if len(sys.argv) <= 1:
	print("usage:  {filename(s)}")
	quit()
infiles = [fname for argv in sys.argv[1:] for fname in glob(argv)]

FIRSTPLOT = True
for fname in infiles:
    assert fname.endswith('_LDAhighestROC.pkl'), fname

    assert '_psiz' in fname
    plottitle = fname[:-len('_LDAhighestROC.pkl')].split('_psiz')[-1]
    psize = int(plottitle.split('_')[0])

    assert '_' in plottitle
    plottitle = plottitle[(plottitle.find('_')+1):]
    assert plottitle.endswith('.txt')
    plottitle = plottitle[:-len('.txt')]

    assert '_expm' in plottitle
    plottitle = plottitle.split('_expm')[0]

    highestroc = cPickle.load(open(fname,'rb'))

    idealdistsfromdiag = (1. - highestroc[2]) / np.sqrt(2.)
    distsfromdiag = (highestroc[3] - highestroc[2]) / np.sqrt(2.)
    maxfromdiag_idx = np.argmax(distsfromdiag)
    maxfromdiag_val = distsfromdiag[maxfromdiag_idx]
    maxfromdiag_string = 'maxfromdiag at fpr '+fstr4(highestroc[2][maxfromdiag_idx])+', tpr '+fstr4(highestroc[3][maxfromdiag_idx])
    print(maxfromdiag_string)

    lw = 2
    if FIRSTPLOT:
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.plot(highestroc[2], highestroc[3], lw=lw, label=plottitle+', AUC '+fstr(highestroc[1])) #, color='darkorange'
    #plt.plot(highestroc[2], distsfromdiag, color='green', label='dists from diag')
    #plt.plot(highestroc[2], idealdistsfromdiag, color='green', linestyle='--', label='ideal dists from diag')

    titlestr = 'Characterizing '+str(psize)+'x'+str(psize)+' patches'
#    if plottitle is not None:
#        titlestr += ', '+str(plottitle)
#    titlestr += '\n' + 'AUC: '+fstr4(highestroc[1]) +'\n'+maxfromdiag_string
    plt.title(titlestr)

    plt.xlabel('false positive rate')
    plt.ylabel('true positive rate')
    plt.xlim([0,1])
    plt.ylim([0,1])
    plt.grid(True)
    plt.axes().set_aspect(1.)
    plt.subplots_adjust(top=0.85)

    #plt.show()
    #plt.savefig(scoresfname+'.png')
    FIRSTPLOT = False

plt.legend(loc='lower right')
plt.show()
