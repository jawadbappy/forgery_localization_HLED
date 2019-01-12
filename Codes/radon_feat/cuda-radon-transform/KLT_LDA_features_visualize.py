"""
Author: Jason Bunk

Visualize features from radon transform processing.
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
import skimage
import skimage.io
import skimage.util
from utils import *
import matplotlib.pyplot as plt
from dataset_medifor_patches import transf_one_patch
from KLT_LDA_features_precompute import process_radon_input

def feats_to_img(feats):
    absfft = np.absolute(feats)
    fftavg = np.mean(absfft, axis=0, keepdims=True) # average across angles
    score = np.absolute(np.divide(feats, fftavg+1e-16))

    return uint8norm(score)# * (score))

def transf_feats(feats, pca_mean, pca_VT, keep_raw=False):
    if len(feats.shape) == 2:
        fresh = feats.reshape((1, -1))
    else:
        assert len(feats.shape) == 3
        fresh = feats.reshape((feats.shape[0], -1))
    fresh = np.dot(fresh - pca_mean, pca_VT.conj().transpose())
    fresh = np.dot(fresh, pca_VT) + pca_mean
    fresh = fresh.reshape(feats.shape)
    if keep_raw:
        return fresh
    return feats_to_img(fresh)

def produce_plots_for_feats(feats, transparms, idxrange):
    '''fig,axs = plt.subplots(2,len(idxrange),figsize=figsize)
    for jj in range(len(idxrange)):
        axs[0,jj].imshow(feats_to_img(feats[idxrange[jj],...]), cmap='gray')
        axs[0,jj].set_title('before PCA')
        axs[1,jj].imshow(transf_feats(feats[idxrange[jj],...], pca_mean, pca_VT), cmap='gray')
        axs[1,jj].set_title('after PCA')
    #fig.suptitle('feats before and after principle comps')
    fig.tight_layout()
    '''

    f2ig,a2xs = plt.subplots(2,len(idxrange),figsize=figsize)
    for jj in range(len(idxrange)):
        thispd = {}
        for key in transparms:
            thispd[key] = transparms[key][idxrange[jj]]
        numnoise = 1
        radoninput = np.zeros(shape=(numnoise, 128, 128, 3), dtype=np.uint8)
        for kk in range(numnoise):
            radoninput[kk,:,:,:] = transf_one_patch( \
                        np.round(npr.uniform(high=255., size=(192,192,3))).astype(np.uint8), thispd)
        noiseradon = np.mean(process_radon_input(radoninput), axis=0)
        assert noiseradon.shape == feats[0,...].shape, str(noiseradon.shape)+" vs "+str(feats[0,...].shape)

        a2xs[0,jj].imshow(feats_to_img(noiseradon), cmap='gray')
        a2xs[0,jj].set_title('noise transformed')

        a2xs[1,jj].imshow(feats_to_img(feats[idxrange[jj],...]), cmap='gray')
        a2xs[1,jj].set_title('from real patch')

        #a2xs[1,jj].imshow(transf_feats(noiseradon, pca_mean, pca_VT), cmap='gray')
        #a2xs[1,jj].set_title('noise w/ PCA')



        #a2xs[0,jj].imshow(feats_to_img(feats[idxrange[jj],...]), cmap='gray')
        #a2xs[1,jj].imshow(transf_feats(feats[idxrange[jj],...], pca_mean, pca_VT), cmap='gray')
    #f2ig.suptitle('feats before and after principle comps')
    f2ig.tight_layout()

if __name__ == '__main__':
    try:
        dfile = sys.argv[1]
    except:
        print("usage:  {file}  {optional:PCA}")
        quit()
    pcafname = None
    if len(sys.argv) > 2:
        pcafname = sys.argv[2]

    if extension_is_image(dfile):
        psize = 96
        img = skimage.io.imread(dfile)
        assert img is not None and img.size > 1 and len(img.shape) == 3, dfile
        wins = skimage.util.view_as_windows(img, (psize, psize, 3), step=128)
        wins = wins.reshape((wins.shape[0], wins.shape[1], wins.shape[3], wins.shape[4], 3))
        print("wins.shape "+str(wins.shape)+", dtype "+str(wins.dtype))

        winsflat = wins.reshape((wins.shape[0]*wins.shape[1], wins.shape[2], wins.shape[3], 3))
        print("winsflat.shape "+str(winsflat.shape)+", dtype "+str(winsflat.dtype))

        RadonPostParams = {'numAngles': 60, 'cropSize': psize}
        winsfeats = process_radon_input(winsflat, RadonPostParams)
        print("winsflatfeats.shape "+str(winsfeats.shape)+", dtype "+str(winsfeats.dtype))

        flatims = np.stack([uint8norm(winsfeats[ii,...]) for ii in range(winsfeats.shape[0])], axis=0)

        print("flatims.shape "+str(flatims.shape)+", dtype "+str(flatims.dtype))

        wins = flatims.reshape((wins.shape[0], wins.shape[1], flatims.shape[1], flatims.shape[2]))
        stackme = np.concatenate([np.concatenate(list(wins[ii,...]),axis=1) for ii in range(wins.shape[0])], axis=0)

        #print("winsfeats.shape "+str(winsfeats.shape)+", dtype "+str(winsfeats.dtype))
        #print("stackme.shape "+str(stackme.shape)+", dtype "+str(stackme.dtype))

        skimage.io.imsave('stacked_features_all_saved.png', stackme)

        quit()


    if dfile.endswith('.'):
        dfile = dfile[:-1]

    pickle_name = dfile.replace('.','_')

    h5file = tables.open_file(pickle_name+'.hdf5',mode='r')
    f0 = h5file.root.f0[:]
    f1 = h5file.root.f1[:]
    h5file.close()

    affjpg_transf_params = cPickle.load(open(pickle_name+'.pkl','rb'))

    BATCH_SIZE = 6
    figsize = (16,9)

    if pcafname is not None and os.path.isfile(pcafname):

        npcacomps = 560

        h5file = tables.open_file(pcafname,mode='r')
        pca_mean = h5file.root.mean[:]
        pca_VT   = h5file.root.VT[:] ##.astype(f0.dtype)
        h5file.close()
        assert len(pca_mean.shape) == 1 and len(pca_VT.shape) == 2 and pca_VT.shape[0] == pca_VT.shape[1], str(pca_mean.shape)+", "+str(pca_VT.shape)

        if False:
            VT_check_1 = np.dot(pca_VT.conj().transpose(), pca_VT)
            VT_check_2 = np.dot(pca_VT, pca_VT.conj().transpose())
            print("err_1_real = "+str(np.mean(np.absolute(np.real(VT_check_1) - np.eye(VT_check_1.shape[0])))))
            print("err_2_real = "+str(np.mean(np.absolute(np.real(VT_check_2) - np.eye(VT_check_2.shape[0])))))
            print("err_1_imag = "+str(np.mean(np.absolute(np.imag(VT_check_1)))))
            print("err_2_imag = "+str(np.mean(np.absolute(np.imag(VT_check_2)))))
            quit()
        if False:
            feats_check = transf_feats(f0, pca_mean, pca_VT, keep_raw=True)
            feats_err = np.absolute(feats_check - f0)
            print("feats_err mean == "+str(np.mean(feats_err)))
            print("feats_err mean-square == "+str(np.mean(np.square(feats_err))))
            print("feats_err amax == "+str(np.amax(feats_err)))
            quit()

        #print("f0.dtype "+str(f0.dtype))
        #print("pca_mean.dtype "+str(pca_mean.dtype))
        #print("pca_VT.dtype "+str(pca_VT.dtype))
        #quit()
        pca_VT = pca_VT[npcacomps:, :]     ##[:npcacomps, :] = 0.

        for ii in range(min(10000,f0.shape[0]//BATCH_SIZE)):
            produce_plots_for_feats(f0, affjpg_transf_params[0], range(ii*BATCH_SIZE, (ii+1)*BATCH_SIZE))
            produce_plots_for_feats(f1, affjpg_transf_params[1], range(ii*BATCH_SIZE, (ii+1)*BATCH_SIZE))
            plt.show()
    else:
        for ii in range(min(10000,f0.shape[0]//BATCH_SIZE)):
            produce_plots_for_feats(f0, affjpg_transf_params[0], range(ii*BATCH_SIZE, (ii+1)*BATCH_SIZE))
            produce_plots_for_feats(f1, affjpg_transf_params[1], range(ii*BATCH_SIZE, (ii+1)*BATCH_SIZE))
            plt.show()
