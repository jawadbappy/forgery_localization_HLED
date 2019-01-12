#!/usr/bin/env python
"""
Author: Jason Bunk

Compute patch feature vector for all patches in an image, and save.
No training, just inference. Requires saved classifiers in a folder (run KLT_LDA_features_classify_after_PCA first).
"""
import os,sys
thispath = os.path.dirname(os.path.abspath(__file__))

os.environ['CUDA_VISIBLE_DEVICES'] = '0'

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

#############################################################
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
#config.gpu_options.per_process_gpu_memory_fraction = 0.3
set_session(tf.Session(config=config))
from keras.models import load_model
#############################################################

import scipy
import scipy.ndimage
import ntpath
from glob import glob

from KLT_LDA_features_precompute import process_radon_input

if __name__ == '__main__':
    try:
        classiffolder = sys.argv[1]
        fraction_manip = float(sys.argv[2])
        out_folder = sys.argv[3]
        assert len(sys.argv) > 4
        assert fraction_manip > 0. and fraction_manip < 1.
    except:
        print("usage:  {folder-with-classifiers}  {fraction-manip}  {output-folder}  {image-filename(s)...}")
        quit()

    assert os.path.exists(classiffolder), classiffolder
    if not os.path.exists(out_folder):
        os.system('mkdir \''+out_folder+'\'')

    if not out_folder.endswith('/'):
        out_folder += '/'
    CLASSIF_FILEEXTN = '_KerasResult.h5'
    CLASSIF_MEANEXTN = '_KerasResult2.h5'

    classifs = [ff for ff in os.listdir(classiffolder) if ff.endswith(CLASSIF_FILEEXTN)]
    infiles = [fname for argv in sys.argv[4:] for fname in glob(argv)]

    patchsize = None
    numradonang = None

    csortme = []
    for cfname in classifs:
        assert cfname.endswith(CLASSIF_FILEEXTN), cfname+'  -- endswith \''+CLASSIF_FILEEXTN+'\''
        cfbase = cfname[:-len(CLASSIF_FILEEXTN)]
        if cfbase.endswith('.txt'):
            cfbase = cfbase[:-len('.txt')]

        assert cfbase.startswith('score_dfft'), cfname
        cfbase = cfbase[len('score_dfft'):]
        doublefftnorm = int(cfbase.split('_')[0])
        cfbase = cfbase[(cfbase.find('_')+1):]

        assert cfbase.startswith('hcf'), cfname
        cfbase = cfbase[len('hcf'):]
        handcraftedfeats = int(cfbase.split('_')[0])
        cfbase = cfbase[(cfbase.find('_')+1):]

        assert '_psiz' in cfbase, cfname
        psizestr = cfbase.split('_psiz')[-1].split('_')[0]
        if patchsize is None:
            patchsize = int(psizestr)
        else:
            assert patchsize == int(psizestr)

        assert cfbase.startswith('a'), cfbase
        angstr = cfbase[1:].split('_')[0]
        if numradonang is None:
            numradonang = int(angstr)
        else:
            assert numradonang == int(angstr)

        laste = cfbase.rfind('expm')
        assert laste > 0 and laste < (len(cfbase)-4)
        cftoe = cfbase[(laste+4):]

        #print("cfname \'"+cfname+"\', cfbase: \'"+cfbase+"\', laste "+str(laste)+", cftoe \'"+cftoe+"\', angstr "+str(angstr))
        #print("doublefftnorm "+str(doublefftnorm)+", handcraftedfeats "+str(handcraftedfeats)+", numradonang "+str(numradonang)+", patchsize "+str(patchsize))
        #quit()

        csortme.append((int(cftoe), os.path.join(classiffolder,cfname)))

    assert patchsize is not None, str(classifs)
    assert numradonang is not None, str(classifs)

    csortme.sort(key=lambda x: x[0])

    print(" ")
    print("num experiments: "+str(len(csortme)))
    print("num angles: "+str(numradonang)+", patch width "+str(patchsize))
    print("num images given: "+str(len(infiles)))
    print(" ")

    ############################################################################
    # load classifiers, set thresholds, then sigmoid

    clfs = []

    for clfnametup in csortme:
        clffile = clfnametup[1]
        assert clffile.endswith(CLASSIF_FILEEXTN)

        clf = load_model(clffile)
        h5file = tables.open_file(clffile[:-len(CLASSIF_FILEEXTN)]+CLASSIF_MEANEXTN,mode='r')
        tr_mean_vec = h5file.root.tr_mean_vec[:]
        tr_scale_vec = h5file.root.tr_scale_vec[:]
        h5file.close()

        assert len(clf.layers) > 1, str(len(clf.layers))
        assert clf.layers[-1].name.startswith('activation_'), str(clf.layers[-1].name)
        assert clf.layers[-2].name.startswith('dense_'), str(clf.layers[-1].name)
        dweights = clf.layers[-2].get_weights()
        assert len(dweights) == 2, len(dweights)
        assert len(dweights[1].shape) == 1 and dweights[1].size == 2, str(dweights[1].shape)
        newbiases = np.zeros_like(dweights[1])
        newbiases[0] = np.log((1. - fraction_manip) / fraction_manip)
        #print("oldbiases "+str(dweights[1])+", newbiases: "+str(newbiases))
        dweights[1] = newbiases
        #print("dweights[-1]: "+str(dweights[1]))
        clf.layers[-2].set_weights(dweights)

        if False:
            print("clf \'"+clffile+"\': len(clf.layers) == "+str(len(clf.layers)))
            for ii in range(len(clf.layers)):
                weightslist = clf.layers[ii].get_weights()
                prstr = "     layer "+str(ii)+": "+str(clf.layers[ii].name)+", len(get_weights()) = "+str(len(weightslist))
                if len(weightslist) > 0:
                    prstr += ": weights shapes: "
                    for jj in range(len(weightslist)):
                        if jj > 0:
                            prstr += ', '
                        prstr += str(weightslist[jj].shape)
                if ii == (len(clf.layers) - 2):
                    assert len(weightslist) == 2
                    assert weightslist[1].size == 2 and len(weightslist[1].shape) == 1
                    print("last layer, biases: "+str(weightslist[1]))

                print(prstr)
            print("\n\n\n")

        #assert clf.priors_.shape == (2,), str(clf.priors_)
        #assert np.sum(np.fabs(clf.priors_ - np.array([0.5,0.5]))) < 1e-9, str(clf.priors_)

        #clf.priors_ = np.array([1.-fraction_manip, fraction_manip])
        clfs.append((clf, tr_mean_vec, tr_scale_vec))

        #clfthreshfile = clffile[:-len('result.hdf5')]+'highestROC.pkl'
        #highestroc = cPickle.load(open(clfthreshfile,'rb'))
        #distsfromdiag = (highestroc[3] - highestroc[2]) / np.sqrt(2.)
        #maxfromdiag_idx = np.argmax(distsfromdiag)

    ############################################################################
    # read and process image(s)

    RadonPostParams = {}
    RadonPostParams['numAngles'] = numradonang
    RadonPostParams['cropSize'] = patchsize
    RadonPostParams['doublefftnorm'] = bool(doublefftnorm)
    RadonPostParams['handcraftedfeats'] = bool(handcraftedfeats)

    fftscore_setup() # setup pyfftw for faster fft

    for imagefname in infiles:

        basename = os.path.join(out_folder, ntpath.basename(imagefname[:-4]))

        try:
            img = cv2.imread(imagefname, cv2.IMREAD_COLOR)
            assert img is not None and img.size > 0, imagefname
            assert len(img.shape) == 3 and img.shape[2] == 3, imagefname
            assert img.dtype == np.uint8
        except:
            print("failed to read \'"+str(img)+"\'")
            continue

        print("\'"+basename+"\': shape "+str(img.shape))
        #cv2.imshow("img",img)
        #cv2.waitKey(0)

        aswindows = view_as_windows(img.copy(), (patchsize, patchsize, 3), step=8)
        pimshape = aswindows.shape
        assert aswindows.dtype == np.uint8, str(aswindows.dtype)
        assert len(pimshape) == 6 and pimshape[2] == 1 and pimshape[5] == 3, str(pimshape)
        assert pimshape[3] == pimshape[4] and pimshape[4] == patchsize, str(pimshape)
        #print("aswindows.shape "+str(pimshape))

        winfeats = process_radon_input( \
                    aswindows.reshape((pimshape[0]*pimshape[1], patchsize, patchsize, 3)).copy(),
                    RadonPostParams)
        assert len(winfeats.shape) == 4
        winfeats = winfeats.reshape((winfeats.shape[0], -1)).copy()

        #print("winfeats.shape "+str(winfeats.shape)+", dtype "+str(winfeats.dtype))

        newfeats = np.zeros((winfeats.shape[0], len(clfs)), dtype=np.float32)

        # see above:
        # each clf == (clf, tr_mean_vec, tr_scale_vec)
        for cidx in range(len(clfs)):
            newfeats[:, cidx] = clfs[cidx][0].predict( \
                                (winfeats-clfs[cidx][1])/clfs[cidx][2], \
                                batch_size=512)[:,1].astype(np.float32)
        newfeats = newfeats.reshape((pimshape[0], pimshape[1], len(clfs))).copy()

        cv2.imwrite(basename+'_clf_0-3.png', uint8_nonorm(newfeats[:,:,:3]))
        cv2.imwrite(basename+'_clf_4-6.png', uint8_nonorm(newfeats[:,:,3:]))
        np.save(basename+'_numpy.npy', newfeats, allow_pickle=False)

        if False:
            newfzoom = scipy.ndimage.interpolation.zoom(newfeats, \
                    (float(img.shape[0])/float(pimshape[0]), \
                     float(img.shape[1])/float(pimshape[1]), 1), order=1, mode='nearest')

            assert newfzoom.shape[0] == img.shape[0], str(newfzoom.shape)+" vs "+str(img.shape)
            assert newfzoom.shape[1] == img.shape[1], str(newfzoom.shape)+" vs "+str(img.shape)

            cv2.imwrite(basename+'_clf_0-3_zoom.png', uint8_nonorm(newfzoom[:,:,:3]))
            cv2.imwrite(basename+'_clf_4-6_zoom.png', uint8_nonorm(newfzoom[:,:,3:]))

        if False:
            assert img.dtype == np.uint8
            img = img.astype(np.float32) / 255.
            segments_slic_RGB = slic(img,      n_segments=128, compactness=10, sigma=1.)
            segments_slic_clf = slic(newfzoom, n_segments=128, compactness=0.25, sigma=4.)

            describe("segments_slic_clf", segments_slic_clf)

            cv2.imwrite(basename+'_seg_RGB.png', uint8_nonorm(mark_boundaries(img, segments_slic_RGB)))
            cv2.imwrite(basename+'_seg_clf.png', uint8_nonorm(mark_boundaries(img, segments_slic_clf)))
            cv2.imwrite(basename+'_seg_clf_0-3.png', uint8_nonorm(mark_boundaries(newfzoom[:,:,:3], segments_slic_clf)))
            cv2.imwrite(basename+'_seg_clf_4-6.png', uint8_nonorm(mark_boundaries(newfzoom[:,:,3:], segments_slic_clf)))
