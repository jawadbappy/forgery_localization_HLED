"""
Author: Jason Bunk

Generate transformations, apply them, and compute radon transform features,
all to a dataset of raw patches.
"""
import os,sys
thispath = os.path.dirname(os.path.abspath(__file__))

from radon_transform_features import radon_transform_features
import time
import numpy as np
import numpy.random as npr
import cv2
import cPickle
import tables
import scipy
import scipy.ndimage
from utils import *
from dataset_medifor_patches import load_hdf5_dataset, new_transf_params, transf_one_patch, update_transf_parms

def process_radon_input(radoninput, RadonPostParams):
    ret = radon_transform_features(radoninput, \
                        numAngles=RadonPostParams['numAngles'], \
                        to3channel=False, doublefftnorm=RadonPostParams['doublefftnorm'], multiprocess=False)
    assert np.isfinite(ret).all()
    return ret

def batch_transf____(patches, transf, RadonPostParams):
    assert len(patches.shape) == 4 and int(patches.shape[1]) == int(patches.shape[2]) and int(patches.shape[3]) == 3
    assert patches.dtype == np.uint8
    nbatch = int(patches.shape[0])

    csize = int(RadonPostParams['cropSize'])

    radoninput = np.zeros(shape=(nbatch, csize, csize, 3), dtype=np.uint8)
    for ii in range(nbatch):
        thispd = {}
        for key in transf:
            thispd[key] = transf[key][ii]
        radoninput[ii,:,:,:] = transf_one_patch(patches[ii,...], thispd, patch_size=csize)
    if bool(RadonPostParams['handcraftedfeats']):
        return process_radon_input(radoninput, RadonPostParams)
    else:
        return radoninput #process_radon_input(radoninput, RadonPostParams)

def transform_patches(patches, transf, RadonPostParams):
    assert len(patches.shape) == 4 and int(patches.shape[1]) == int(patches.shape[2]) and int(patches.shape[3]) == 3
    assert patches.dtype == np.uint8

    nbatch = int(patches.shape[0])
    ngroup = 2000

    if nbatch < ngroup:
        return batch_transf____(patches, transf, RadonPostParams)

    getsize = batch_transf____(np.expand_dims(patches[0,...],0), transf, RadonPostParams)
    assert int(getsize.shape[0]) == 1
    all_radon_out = np.zeros([nbatch,]+list(getsize.shape[1:]), dtype=getsize.dtype)

    nloops = nbatch // ngroup
    for ii in range(nloops):
        thispd = {}
        for key in transf:
            thispd[key] = transf[key][(ii*ngroup):((ii+1)*ngroup)]
        all_radon_out[(ii*ngroup):((ii+1)*ngroup),...] = batch_transf____(patches[(ii*ngroup):((ii+1)*ngroup),...].copy(), thispd, RadonPostParams)
        print("done with "+str((ii+1)*ngroup)+" / "+str(nbatch))

    return all_radon_out


if __name__ == '__main__':
    try:
        indfilepath    = sys.argv[1]
        outputbasename = sys.argv[2]
        numradonang    = int(sys.argv[3])
        radonpatchsize = int(sys.argv[4])
        doublefftnorm  = int(sys.argv[5])
        handcraftedfeats=int(sys.argv[6])
    except:
        print("usage:  {input-patches}  {output-basename}  {num-radon-angles}  {patch-size}  {doublefft}  {handcraftedfeats}")
        quit()
    radon_post_parms = {}
    radon_post_parms['numAngles'] = numradonang
    radon_post_parms['cropSize'] = radonpatchsize
    radon_post_parms['doublefftnorm'] = bool(doublefftnorm)
    radon_post_parms['handcraftedfeats'] = bool(handcraftedfeats)

    trainX = load_hdf5_dataset(indfilepath) #[:100,...]
    print("KLT_LDA_features_precompute: trainX.shape "+str(trainX.shape))
    npr.shuffle(trainX) # shuffle along first axis

    fftscore_setup() # setup pyfftw for faster fft

    transfs = []
    feats = []

    if len(trainX.shape) == 4:
        halfnumpatches = int(trainX.shape[0]) // 2
        if not (halfnumpatches*2) == int(trainX.shape[0]):
            halfnumpatches += 1
        #assert int(trainX.shape[0]) % 2 == 0 and (halfnumpatches*2) == int(trainX.shape[0]), str(trainX.shape)

        for ii in range(2):
            if ii == 0:
                patches = trainX[:halfnumpatches, ...]
                update_transf_parms('transform_parms_baseline.txt')
            else:
                patches = trainX[halfnumpatches:, ...]
                update_transf_parms('transform_parms_detectme.txt')

            transfs.append(new_transf_params(patches.shape[0], None))
            feats.append(transform_patches(patches, transfs[-1], radon_post_parms))
            print("feats[-1].shape "+str(feats[-1].shape)+", feats[-1].dtype "+str(feats[-1].dtype))

    elif len(trainX.shape) == 5:
        npatchper = 9
        assert npatchper <= (int(trainX.shape[1]) // 2)
        for ii in range(2):
            patches = trainX[:, (ii*npatchper):((ii+1)*npatchper), ...]
            patches = patches.reshape((patches.shape[0]*patches.shape[1], patches.shape[2], patches.shape[3], patches.shape[4])).copy()

            if ii == 0:
                update_transf_parms('transform_parms_baseline.txt')
            elif ii == 1:
                update_transf_parms('transform_parms_detectme.txt')
            else:
                assert 0, 'ii: '+str(ii)

            transfs.append(new_transf_params(patches.shape[0], None))
            feats.append(transform_patches(patches, transfs[-1], radon_post_parms))
            print("feats[-1].shape "+str(feats[-1].shape)+", feats[-1].dtype "+str(feats[-1].dtype))

    h5file = tables.open_file(outputbasename+'.hdf5',mode='w')
    for ii in range(2):
        h5file.create_array(h5file.root, 'f'+str(ii), feats[ii])
    h5file.close()
    cPickle.dump(transfs, open(outputbasename+'.pkl','wb'))
    print("DONE")
