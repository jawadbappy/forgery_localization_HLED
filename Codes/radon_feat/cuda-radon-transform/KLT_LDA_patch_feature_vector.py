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
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from skimage.segmentation import mark_boundaries
from skimage.segmentation import slic
import scipy
import scipy.ndimage
import ntpath
from glob import glob

from KLT_LDA_features_precompute import process_radon_input
from KLT_LDA_features_classify_after_PCA import my_quadratic_log_probs

if __name__ == '__main__':
    try:
        classiffolder = sys.argv[1]
        fraction_manip = float(sys.argv[2])
        assert len(sys.argv) > 3
        assert fraction_manip > 0. and fraction_manip < 1.
    except:
        print("usage:  {folder-with-classifiers}  {fraction-manip}  {image-filename(s)...}")
        quit()

    OUTPUT_FOLDER = os.path.join(thispath, 'output')
    if not OUTPUT_FOLDER.endswith('/'):
        OUTPUT_FOLDER += '/'

    classifs = [ff for ff in os.listdir(classiffolder) if ff.endswith('_LDAresult.hdf5')]
    infiles = [fname for argv in sys.argv[3:] for fname in glob(argv)]

    patchsize = None
    numradonang = None

    csortme = []
    for cfname in classifs:
        assert cfname.endswith('_LDAresult.hdf5')
        cfbase = cfname[:-len('_LDAresult.hdf5')]
        if cfbase.endswith('.txt'):
            cfbase = cfbase[:-len('.txt')]

        assert '_psiz' in cfbase, cfname
        psizestr = cfbase.split('_psiz')[-1].split('_')[0]
        if patchsize is None:
            patchsize = int(psizestr)
        else:
            assert patchsize == int(psizestr)

        assert cfbase.startswith('scores__a'), cfname
        angstr = cfbase[len('scores__a'):].split('_')[0]
        if numradonang is None:
            numradonang = int(angstr)
        else:
            assert numradonang == int(angstr)

        laste = cfbase.rfind('expm')
        assert laste > 0 and laste < (len(cfbase)-4)
        cftoe = cfbase[(laste+4):]

        #print("cfname \'"+cfname+"\', cfbase: \'"+cfbase+"\', laste "+str(laste)+", cftoe \'"+cftoe+"\', angstr "+str(angstr))
        #quit()

        csortme.append((int(cftoe), os.path.join(classiffolder,cfname)))

    csortme.sort(key=lambda x: x[0])

    print(" ")
    print("num experiments: "+str(len(csortme)))
    print("num angles: "+str(numradonang)+", patch width "+str(patchsize))
    print(" ")

    ############################################################################
    # load classifiers, set thresholds, then sigmoid

    clfs = []

    for clfnametup in csortme:
        clffile = clfnametup[1]
        assert clffile.endswith('_LDAresult.hdf5')

        clf = QuadraticDiscriminantAnalysis(store_covariances=False)
        clf.classes_ = np.array([0,1],dtype=np.int64)
        h5file = tables.open_file(clffile,mode='r')
        clf.means_ = np.array(h5file.root.means[:])
        clf.priors_ = np.array(h5file.root.priors[:])
        clf.rotations_ = np.array(h5file.root.rotations[:])
        clf.scalings_ = np.array(h5file.root.scalings[:])
        h5file.close()

        assert clf.priors_.shape == (2,), str(clf.priors_)
        assert np.sum(np.fabs(clf.priors_ - np.array([0.5,0.5]))) < 1e-9, str(clf.priors_)

        clf.priors_ = np.array([1.-fraction_manip, fraction_manip])
        clfs.append(clf)

        #clfthreshfile = clffile[:-len('result.hdf5')]+'highestROC.pkl'
        #highestroc = cPickle.load(open(clfthreshfile,'rb'))
        #distsfromdiag = (highestroc[3] - highestroc[2]) / np.sqrt(2.)
        #maxfromdiag_idx = np.argmax(distsfromdiag)

    ############################################################################
    # read and process image(s)

    RadonPostParams = {}
    RadonPostParams['numAngles'] = numradonang
    RadonPostParams['cropSize'] = patchsize

    for imagefname in infiles:

        basename = os.path.join(OUTPUT_FOLDER, ntpath.basename(imagefname[:-4]))

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
        assert len(winfeats.shape) == 3
        winfeats = winfeats.reshape((winfeats.shape[0], winfeats.shape[1]*winfeats.shape[2])).copy()

        #print("winfeats.shape "+str(winfeats.shape)+", dtype "+str(winfeats.dtype))

        newfeats = np.zeros((winfeats.shape[0], len(clfs)), dtype=np.float32)

        for cidx in range(len(clfs)):
            newfeats[:, cidx] = np.exp(my_quadratic_log_probs(clfs[cidx], winfeats)[:,1]).astype(np.float32)
        newfeats = newfeats.reshape((pimshape[0], pimshape[1], len(clfs))).copy()

        cv2.imwrite(basename+'_clf_0-3.png', uint8_nonorm(newfeats[:,:,:3]))
        cv2.imwrite(basename+'_clf_4-6.png', uint8_nonorm(newfeats[:,:,3:]))

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
