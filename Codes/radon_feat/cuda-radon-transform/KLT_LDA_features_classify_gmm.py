"""
Author: Jason Bunk

Experiment with a Bayesian GMM classifier.
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
from sklearn.mixture import BayesianGaussianMixture

from KLT_LDA_features_classify_after_PCA import load_hdf5_train_set


def my_log_softmax(X):
    assert len(X.shape) == 2
    xmax = np.amax(X, axis=1, keepdims=True)
    log_softmax_denom = xmax + np.log(np.sum(np.exp(X - xmax), axis=1,keepdims=True))
    return X - log_softmax_denom

def load_hdf5_train_set_keep_split(filename):
    h5file = tables.open_file(filename,mode='r')
    f0 = h5file.root.f0[:] #2000,...]
    f1 = h5file.root.f1[:] #2000,...]
    h5file.close()
    return f0.reshape((f0.shape[0],-1)), \
           f1.reshape((f1.shape[0],-1))


if __name__ == '__main__':
    try:
        traindatafilename = sys.argv[1]
    except:
        print("usage:   {train-set-filename}   {optional:valid-set-filename}   {optional:scores-out-filename}")
        quit()
    validfilename = None
    scoresfname = None
    if len(sys.argv) > 2:
        validfilename = sys.argv[2]
    if len(sys.argv) > 3:
        scoresfname = sys.argv[3]
    assert scoresfname is not None

    while traindatafilename.endswith('.'):
        traindatafilename = traindatafilename[:-1]
    while validfilename.endswith('.'):
        validfilename = validfilename[:-1]

    tr_0, tr_1 = load_hdf5_train_set_keep_split(traindatafilename+'.hdf5')
    va_X, va_Y = load_hdf5_train_set(validfilename+'.hdf5')

    describe("BEFORE MEAN SUBTR: tr_0", tr_0)
    describe("BEFORE MEAN SUBTR: tr_1", tr_1)

    for ncomps in [2,3,4,5,6]:
        build_GMM = lambda trset: BayesianGaussianMixture(n_components=ncomps, \
                                max_iter=200, n_init=3, \
                                covariance_type='full', \
                                weight_concentration_prior_type='dirichlet_distribution', \
                                verbose=2).fit(trset)

        beftime = time.time()
        clf_0 = build_GMM(tr_0)
        clf_1 = build_GMM(tr_1)
        afttime = time.time()
        print("fitting took "+str(afttime - beftime)+" sec")

        LL_0 = clf_0.score_samples(va_X)
        LL_1 = clf_1.score_samples(va_X)

        assert len(LL_0.shape) == 1 and len(LL_1.shape) == 1
        LL_both = np.stack((LL_0, LL_1), axis=-1)
        assert len(LL_both.shape) == 2 and LL_both.shape[1] == 2

        preds = np.exp(my_log_softmax(LL_both)[:, 1])

        fpr, tpr, thresholds = roc_curve(va_Y, preds)
        roc_auc = auc(fpr, tpr)
        distsfromdiag = (tpr - fpr) / np.sqrt(2.)
        maxfromdiag_idx = np.argmax(distsfromdiag)
        maxfromdiag_val = distsfromdiag[maxfromdiag_idx]
        assert np.fabs(maxfromdiag_val - np.amax(distsfromdiag)) < 1e-12

        # make plot
        highestroc = (0., roc_auc, fpr, tpr)

        idealdistsfromdiag = (1. - highestroc[2]) / np.sqrt(2.)
        distsfromdiag = (highestroc[3] - highestroc[2]) / np.sqrt(2.)
        maxfromdiag_idx = np.argmax(distsfromdiag)
        maxfromdiag_val = distsfromdiag[maxfromdiag_idx]
        maxfromdiag_string = 'maxfromdiag at fpr '+fstr4(highestroc[2][maxfromdiag_idx])+', tpr '+fstr4(highestroc[3][maxfromdiag_idx])
        print(maxfromdiag_string)

        lw = 2
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.plot(highestroc[2], highestroc[3], color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % highestroc[1])
        plt.plot(highestroc[2], distsfromdiag, color='green', label='dists from diag')
        plt.plot(highestroc[2], idealdistsfromdiag, color='green', linestyle='--', label='ideal dists from diag')
        titlestr = 'Classifying patches w/ GMM w/ '+str(ncomps)+' components\n' \
                + 'AUC: '+fstr4(highestroc[1])+'\n'+maxfromdiag_string
        plt.title(titlestr)
        plt.xlabel('false positive rate')
        plt.ylabel('true positive rate')
        plt.xlim([0,1])
        plt.ylim([0,1])
        plt.grid(True)
        plt.axes().set_aspect(1.)
        plt.subplots_adjust(top=0.85)
        #plt.tight_layout()
        plt.savefig(scoresfname+'_ncomp'+str(ncomps)+'.png')
        #plt.show()
