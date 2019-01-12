"""
Author: Jason Bunk

Compute patch feature vector for all patches in an image, and save.
No training, just inference. Requires saved classifiers in a folder (run KLT_LDA_features_classify_after_PCA first).
"""
import os,sys
thispath = os.path.dirname(os.path.abspath(__file__))

import time, gc
import numpy as np
import numpy.random as npr
import cv2
import cPickle
import tables
from utils import *
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import roc_curve, auc
from skimage.util import view_as_windows

import scipy
import scipy.ndimage
import ntpath
from glob import glob
from natsort import natsorted

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        print("usage:  {directories-with-same-images-but-different-classifiers}")
        quit()

    indirs = [fname for argv in sys.argv[1:] for fname in glob(argv)]
    infiles = []
    for ii in range(len(indirs)):
        idir = indirs[ii]
        infiles.append(natsorted([ff for ff in os.listdir(idir) if ff[-4:] in ['.png', '.npy']]))
        if len(infiles) > 1:
            assert infiles[-1] == infiles[-2], 'must have matching images\n'+str(infiles[-1])+'\n'+str(infiles[-2])

    imgs = {}
    for ii in range(len(indirs)):
        idir = indirs[ii]
        for jj in range(len(infiles[ii])):
            fname = infiles[ii][jj]
            fpath = os.path.join(idir, fname)
            assert os.path.isfile(fpath), fpath
            if fpath.endswith('.npy'):
                im = np.load(fpath).astype(np.float32)
            else:
                im = cv2.imread(fpath, cv2.IMREAD_COLOR).astype(np.float32)
            assert im is not None and im.size > 1 and len(im.shape) == 3
            if fname not in imgs:
                imgs[fname] = [im,]
            else:
                imgs[fname] += [im,] # append

    for key in imgs:
        print(key+" is average of "+str(len(imgs[key]))+" images")
        imgs[key] = np.mean(np.stack(imgs[key],axis=0),axis=0)

    for key in imgs:
        outfname = os.path.join('output',key)
        if key.endswith('.npy'):
            np.save(outfname, imgs[key], allow_pickle=False)
        else:
            cv2.imwrite(outfname, imgs[key])
