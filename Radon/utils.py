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

