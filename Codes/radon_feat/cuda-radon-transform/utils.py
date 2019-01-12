"""
Author: Jason Bunk

Misc. utils
"""
import time
import numpy as np
import pyfftw

def describe(name,arr):
	try:
		print(name+", shape: "+str(arr.shape)+", dtype: "+str(arr.dtype) \
				+" (min,max): ("+str(np.amin(arr))+", "+str(np.amax(arr))+")" \
				+" (mean,std): ("+str(np.mean(arr))+", "+str(np.std(arr))+")")
	except:
		try:
			print(name+", shape "+str(arr.get_shape())+", type: "+str(type(arr)))
		except:
			print(name+", type: "+str(type(arr)))

def extension_is_image(fname):
	ff = fname.lower()
	return ff.endswith('.jpg') or ff.endswith('.jpeg') or ff.endswith('.png') \
		or ff.endswith('.tif') or ff.endswith('.pgm')  or ff.endswith('.ppm') \
		or ff.endswith('.bmp')

def saveandshow(wname, img):
	import cv2
	cv2.imwrite(wname+'.png', img)
	cv2.imshow(wname, img)

def imshow_nottoobig(wname, img):
	import cv2
	fsc = np.sqrt(float(1280*720) / float(img.shape[0]*img.shape[1]))
	if fsc < 1.:
		cv2.imshow(wname, cv2.resize(img, (0,0), fx=fsc, fy=fsc, interpolation=cv2.INTER_AREA))
	else:
		cv2.imshow(wname, img)

def mynorm(arr):
	amin = np.amin(arr)
	return np.absolute((arr-amin)/(np.amax(arr)-amin+1e-18))

def uint8_nonorm(arr):
	return np.round(arr*255.).astype(np.uint8)

def uint8norm(arr):
	return np.round(mynorm(arr)*255.0).astype(np.uint8)

def centerednorm(arr):
	amed = np.median(arr, axis=[0,1], keepdims=True)
	ret = arr - amed
	ascale = np.sqrt(np.mean(np.fabs(ret)**2))
	return ret/ascale

def centered_uint8_norm(arr, scale=10.0):
	return np.round(np.clip(centerednorm(arr)*scale+127.5, a_min=0, a_max=255)).astype(np.uint8)

def to2chanRGB(im0, im1):
	assert len(im0.shape) == 2 and len(im1.shape) == 2
	assert im0.shape == im1.shape
	im2 = np.zeros_like(im0)
	return uint8norm(np.stack((im2, im1, im0), axis=2))

def complex_to_interleaved_real(arr, axis):
	axis = int(axis)
	realpart = np.real(arr)
	imagpart = np.imag(arr)
	if arr.dtype == np.float32 or arr.dtype == np.float64:
		assert np.sum(np.fabs(np.absolute(imagpart))) < 1e-20
		return arr
	newshape = list(arr.shape)
	newshape[axis] *= 2
	ret = np.zeros(newshape, dtype=np.float32)
	if axis == 0:
		ret[0::2, ...] = realpart
		ret[1::2, ...] = imagpart
	elif axis == 1:
		if len(arr.shape) == 2:
			ret[:, 0::2] = realpart
			ret[:, 1::2] = imagpart
		else:
			ret[:, 0::2, ...] = realpart
			ret[:, 1::2, ...] = imagpart
	else:
		assert 0
	return ret

def fftscore_setup():
	pyfftw.interfaces.cache.enable()
	pyfftw.interfaces.cache.set_keepalive_time(8.0)

def fftscores(arrs, is_already_fft=False):
	if len(arrs.shape) == 2:
		print("fftscores: warning: arrs should be in batch mode, with the 0-axis indexing batch items")
		arrs = arrs.reshape([1,]+list(arrs.shape))
	if is_already_fft: # the CUDA code can be set up to do the fft, but it doesn't always work,
		absfft = arrs.copy() # cuFFT requires the array sizes to be nicely factorable
	else:
		#truefft = np.fft.rfft(arrs, axis=2)
		truefft = pyfftw.interfaces.numpy_fft.rfft(arrs, axis=2, planner_effort='FFTW_PATIENT', threads=6)
		absfft = np.absolute(truefft)
	fftmax = np.amax(absfft, axis=1, keepdims=True) # max across angles
	fftavg = np.mean(absfft, axis=1, keepdims=True) # average across angles

	if True:
		#print("fftscore is complex")
		score = np.divide(truefft, fftavg+1e-16)
	else:
		print("fftscore is abs-based")
		score = np.divide(absfft, fftavg+1e-16)

	return absfft, score, np.divide(fftmax, fftavg+1e-16), fftavg

def plotfftscores(basename, scorestuple):
	import matplotlib.pyplot as plt
	absfft, fftnormed, score, fftavg = scorestuple
	fftnormed = np.absolute(fftnormed)
	assert len(absfft.shape) == 3 and len(fftnormed.shape) == 3 and len(score.shape) == 3
	assert fftnormed.shape == absfft.shape and score.shape[1] == 1
	nbatch = absfft.shape[0]
	#misc.imsave(basename+'_fft.png', uint8norm(absfft))
	fig, axes = plt.subplots(nbatch, 2)#, figsize=(8, 4.5))
	if len(axes.shape) == 1:
		axes = axes.reshape([1,]+list(axes.shape))
	for bidx in range(nbatch):
		axes[bidx,0].imshow(fftnormed[bidx,:,:])
		axes[bidx,0].set_title(basename+"\nfft normed")
		axes[bidx,0].set_xlabel('sensor')
		axes[bidx,0].set_ylabel('angle')
		axes[bidx,1].plot(score[bidx,0,:])
		axes[bidx,1].set_title(basename+"\nfft detector score")

def shuffle_in_unison(a, b):
    p = np.random.permutation(len(a))
    if b is None:
        return a[p], None
    assert len(a) == len(b)
    return a[p], b[p]

def fstr(fvar):
    return "{0:.3g}".format(fvar)
def fstr4(fvar):
    return "{0:.4g}".format(fvar)

class precision_recall_finder:
    def __init__(self):
        self.reset()
    def reset(self):
        self.truepos = 0.0
        self.trueneg = 0.0
        self.falsepos = 0.0
        self.falseneg = 0.0
    def update(self, preds, truth):
        self.truepos  += np.sum(np.multiply(preds, truth))
        self.trueneg  += np.sum(np.multiply(1.0 - preds, 1.0 - truth))
        self.falsepos += np.sum(np.multiply(preds, 1.0 - truth))
        self.falseneg += np.sum(np.multiply(1.0 - preds, truth))
    def report(self, reset=False):
        precision = self.truepos / (self.truepos + self.falsepos + 1e-9)
        recall    = self.truepos / (self.truepos + self.falseneg + 1e-9)
        accuracy = (self.truepos + self.trueneg) / (self.truepos + self.trueneg + self.falsepos + self.falseneg + 1e-9)
        if reset:
            self.reset()
        return precision, recall, 2.0*precision*recall/(precision+recall+1e-9), accuracy

def concat_batch(imlist, axis, normalization=None):
	if axis == 0:
		pxis = 1
	elif axis == 1:
		pxis = 0
	else:
		assert 0
	maxsize = max([im.shape[pxis] for im in imlist])
	padims = []
	for im in imlist:
		if im.shape[pxis] == maxsize:
			padims.append(im)
		else:
			padarr = [[0,0] for ii in range(len(im.shape))]
			padarr[pxis][0] = (maxsize - im.shape[pxis]) // 2
			padarr[pxis][1] = (maxsize - im.shape[pxis] - padarr[pxis][0])
			padims.append(np.pad(im,padarr,mode='constant'))
			assert padims[-1].shape[pxis] == maxsize, str(padims[-1].shape[pxis])+" != "+str(maxsize)+", im.shape[pxis] "+str(im.shape[pxis])+", padarr: "+str(padarr)
	if normalization is not None:
		padims = [normalization(pp) for pp in padims]
	conc = np.concatenate(padims,axis=axis)
	if conc.dtype == np.uint8:
		return conc
	elif conc.dtype == np.float32:
		#if np.amax(conc) > 1.0:
		return np.round(conc).astype(np.uint8)
		#else:
		#	return np.round(conc*255.0).astype(np.uint8)
	assert 0 # else assert 0

def concat_and_show(wname,imlist,axis):
	import cv2
	cv2.imshow(wname, concat_batch(imlist, axis))
