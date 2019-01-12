"""
Author: Jason Bunk

This file contains an interface to compute features on image patch(es),
using radon_transform_features().
To import from "pysinogram", the C++ and CUDA code must be built.

When feeding these features into another CUDA framework like Tensorflow,
it is recommended to use the "multiprocess=True" option,
for best compatibility / stability.
"""
import os,sys
thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(thispath,'build'))
import multiprocessing
import numpy as np

from pysinogram import BatchRadonTransform

from utils import *


def radon_transform_features_____(patches, numAngles, to3channel, verbose, doublefftnorm):
    numAngles = int(numAngles)
    to3channel = bool(to3channel)
    assert doublefftnorm is not None
    assert len(patches.shape) == 4, "Please provide a batch of patches, in a numpy array of shape [batchsize, width, width, 3]."
    assert patches.shape[1] == patches.shape[2], "Patches must be square."
    assert patches.shape[3] == 3, "Patches must be RGB."

    theta = np.linspace(0,180,numAngles,endpoint=False)
    # circle_inscribed == False == 0, do_laplacian_first == True == 1
    batch = np.array(BatchRadonTransform(list(patches), list(theta), 0, 1))

    if verbose:
        describe("batch (after radon)", np.array(batch))
        beftime = time.time()

    if doublefftnorm:
        # concatenate normalized FFT with the normalization factor (fftavg)
        #           fftavg is the 1D FFTs average over all angles;
        #  assuming typical 2D FFT spectrum will be mostly circularly symmetric,
        #  while resampling signals won't be (will only show up every 90 degrees),
        #  so the normalization enhances the irregularities.
        #  Concatenating with the average ensures the normalization does not throw away information.
        absfft, fftnormed, score, fftavg = fftscores(batch)
        batch = np.expand_dims(np.concatenate([np.absolute(fftnormed), fftavg], axis=1), axis=-1)
    else:
        batch = np.expand_dims(fftscores(batch)[1], axis=-1)

    if verbose:
        print("fft took "+str(time.time() - beftime)+" seconds")
        describe("batch (after fft)", batch)

    assert len(batch.shape) == 4 and batch.shape[-1] in [1,2]
    if to3channel:
        assert 0
        bshape = list(batch.shape)+[3,]
        ret = np.zeros(bshape, dtype=np.float32)
        ret[:,:,:,0] = np.real(batch)
        ret[:,:,:,1] = np.imag(batch)
        ret[:,:,:,2] = np.absolute(batch) - 1.
        return ret
    else:
        return batch


def multiprocess_feature_computer(child_p, numAngles, to3channel, doublefftnorm):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    print("radon_transform_features: set CUDA_VISIBLE_DEVICES to "+str(os.environ['CUDA_VISIBLE_DEVICES']))
    while True:
        try:
            patches = child_p.recv() # Read from the output pipe
        except EOFError: # pipe closed
            print("Pipe Child: closed")
            return
        if type(patches) == type('kill') and patches == 'kill':
            print("Pipe Child: closed")
            return
        #describe("Pipe Child: recv patches", patches)
        ret = radon_transform_features_____(patches, numAngles, to3channel, verbose=False, doublefftnorm=doublefftnorm)
        #describe("Pipe Child: send ret", ret)
        child_p.send(ret)

childprocess = None
def radon_transform_features(patches, numAngles=360, to3channel=False, doublefftnorm=None, verbose=False, multiprocess=False):
    global childprocess
    if not multiprocess:
        return radon_transform_features_____(patches, numAngles, to3channel, verbose, doublefftnorm=doublefftnorm)
    else:
        if childprocess is None:
            print("### creating new process for radon transform to be computed on GPU 1")
            parent_p, child_p = multiprocessing.Pipe()
            reader = multiprocessing.Process(target=multiprocess_feature_computer, \
                        args=(child_p, numAngles, to3channel, doublefftnorm))
            reader.start() # Launch the reader process

            childprocess = (reader, parent_p, child_p)
        else:
            reader, parent_p, child_p = childprocess

        #describe("Pipe Parnt: send patches", patches)

        parent_p.send(patches)
        return parent_p.recv()

        #describe("Pipe Parnt: recv ret", ret)
        #quit()
        #return ret

def radon_transform_features_kill_child_process():
    global childprocess
    if childprocess is not None:
        reader, parent_p, child_p = childprocess
        parent_p.send('kill')
        reader.join()
