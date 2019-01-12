"""
Author: Jason Bunk

Unsupervised dimensionality reduction:
reduce dimensionality of precomputed features.
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

def processfeats(feats):
    return feats ######np.absolute(feats)
def feats2image(img):
    return uint8norm(np.absolute(img))

if __name__ == '__main__':
    try:
        dfile = sys.argv[1]
    except:
        print("usage:  {file}")
        quit()
    if dfile.endswith('.'):
        dfile = dfile[:-1]

    ncomponents = -1

    pickle_name = dfile.replace('.','_')

    h5file = tables.open_file(pickle_name+'.hdf5',mode='r')
    f0 = h5file.root.f0[:]
    f1 = h5file.root.f1[:]
    h5file.close()

    ndataset = int(f0.shape[0])
    f0shape = f0.shape
    print("f0shape == "+str(f0shape))
    assert int(f0.shape[0]) == int(f1.shape[0])

    stacked = processfeats(np.concatenate((f0,f1), axis=0))
    stacked = stacked.reshape((stacked.shape[0],-1))
    del f0
    del f1
    gc.collect()
    print("stacked.shape "+str(stacked.shape)+", stacked.dtype "+str(stacked.dtype))

    assert len(stacked.shape) == 2
    if ncomponents < 0:
        ncomponents = min(stacked.shape[0], stacked.shape[1])

    from sklearn.utils.extmath import randomized_svd
    orig_stack = stacked.copy()
    stackemean = np.mean(stacked, axis=0)
    stacked -= stackemean
    print("starting SVD")
    U,s,VT = randomized_svd(stacked, n_components=ncomponents)#, n_iter=6)
    print("SVD done! VT.shape "+str(VT.shape)+", VT.dtype "+str(VT.dtype))

    h5file = tables.open_file(pickle_name+'_PCA_transfparams.hdf5',mode='w')
    h5file.create_array(h5file.root, 'mean', stackemean)
    h5file.create_array(h5file.root, 'VT', VT)
    h5file.create_array(h5file.root, 's', s)
    h5file.close()
    print("SAVED TRANSFORM PARAMS")

    print("VT.shape "+str(VT.shape))

    #h5file = tables.open_file(pickle_name+'_PCA.hdf5',mode='w')
    #for ii in range(2):
    #    h5file.create_array(h5file.root, 'f'+str(ii), stacktransf[(ii*ndataset):((ii+1)*ndataset),...])
    #h5file.close()

    # verbose
    s = np.absolute(s)
    s /= np.sum(s)
    print("eigenvalues: "+str(s))
    print("(number of eigenvalues: "+str(s.size)+")")
    plt.plot(np.cumsum(s))
    plt.title('cumulative sum of eigenvalues')
    plt.show()

    ngroup = min(10, VT.shape[0])
    for ii in range(VT.shape[0] // ngroup):
        showthis = np.zeros((f0shape[1], f0shape[2] * ngroup), dtype=np.uint8)
        for jj in range(ngroup):
            eigidx = ii*ngroup+jj
            vresh = feats2image(VT[eigidx,...].reshape(f0shape[1:]))
            showthis[:, (jj*f0shape[2]):((jj+1)*f0shape[2])] = vresh
            print("eigenvector "+str(eigidx)+" with variance fraction "+str(s[eigidx]))
        cv2.imshow("eigenvectors", showthis)
        cv2.waitKey(0)

    for jj in range(10):
        partialV = VT[:(jj*100), :]

        stacktransf = np.dot(stacked,partialV.transpose())
        print("AFTER transformation, stacktransf.shape "+str(stacktransf.shape))

        reconstr_stack = np.dot(stacktransf, partialV) + stackemean
        assert reconstr_stack.shape == orig_stack.shape
        rec_error = np.square(np.absolute(orig_stack - reconstr_stack))
        rec_mean_err = np.mean(rec_error, axis=0)
        print("mean error: "+str(np.mean(rec_mean_err)))
        cv2.imshow("mean_err", feats2image(rec_mean_err.reshape(f0shape[1:])))
        cv2.waitKey(0)
    #quit()
