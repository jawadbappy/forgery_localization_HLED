"""
Author: Jason Bunk, modified by @Jawad Bappy

Test CUDA radon transform by comparing the output to skimage.transform.radon()
which does the same thing.

Since the CUDA radon code now also does 3x3 laplacian filter before the sinogram,
test that also using scipy.signal.convolve2d()

scipy.signal.convolve2d() and CUDA convolutionNotSeparableGPU() in convolutionTexture.cu
		produce exactly identical outputs, down to floating point precision.

skimage.transform.radon() and the CUDA code in sinogram.cu produce extremely similar
outputs, but not identical, perhaps due to a difference in interpolation.
The difference is negligible and it does NOT have a significant effect
                                    on subsequent feature extraction using FFT.
"""
import os,sys

import numpy as np
from scipy import signal
import time
import skimage
import skimage.io, skimage.transform
from skimage.transform import resize
from skimage.util import view_as_windows
import scipy.misc
import scipy.io as sio
from utils import *
from skimage import img_as_uint
#imagefname = 'purple_bus_rbcfbeb24t_crop.png'
#imagefpath = os.path.join(thispath,imagefname)

# load image file
root='./test/evalset/manip/'
destination='/home/mbapp001/MediFor/Radon/test/resamp_feat/'

imageList=os.listdir(root)
print len(imageList)

for q in range(0,len(imageList)):
	imName=root+'/'+imageList[q]
	if q==204:
		continue
	if imName.endswith('.jpg'):
		im=skimage.io.imread(imName)
	else:
		continue		
	
	orig_object=im
	#orig_object = skimage.io.imread(imagefpath)
	print q
	if len(orig_object.shape)!=3: 
		print 'Not a Valid Image!!'
		continue
	assert orig_object is not None and orig_object.size > 1 and len(orig_object.shape) == 3, imName
	
	#orig_object=resize(im,(1024,1024,3),mode='reflect').astype(np.uint8)
	#orig_object.astype(np.uint8)
	orig_object=scipy.misc.imresize(im,(64,64,3),interp='bilinear',mode=None)
	im_test=orig_object
	# extract square patches with stride (step) 8
	patchsize=64
	# reshape to a list of patches
	listofpatches = orig_object.reshape((1, patchsize, patchsize, 3))
	print("patches array reshaped to list of patches with shape "+str(listofpatches.shape))

	# Radon projection parameters
	circle_inscribed = False
	numAngles = 18
	theta = np.linspace(0,180,numAngles,endpoint=False)

	def radon_projections_compiled_cuda(patches, thetas, circle_inscribed):
		sys.path.append(os.path.join(thispath,'build')) # for importing pysinogram.so
		from pysinogram import BatchRadonTransform
		return np.array(BatchRadonTransform(list(patches), list(thetas), circle_inscribed))

	def radon_projections_skimage_python(patches, thetas, circle_inscribed):
		# sqrt(abs(  2D discrete 3x3 laplacian filter  ))
		kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
		laplacefilter = lambda xx: np.sqrt(np.fabs(signal.convolve2d(xx, kernel, mode='same', boundary='symm')))
		# do laplacian filter on each channel independently, then average across channels
		rgbfilter = lambda xx: np.mean([laplacefilter(xx[:,:,chan]) for chan in range(xx.shape[2])], axis=0)
		# transpose the sinogram output, to be consistent with CUDA implementation above
		myradon = lambda xx: skimage.transform.radon(rgbfilter(xx), theta=theta, circle=circle_inscribed).transpose()
		# iterate processing over all patches
		return np.stack([myradon(patches[ii,...]) for ii in range(patches.shape[0])], axis=0)

	# run tests
	if False:
		# compare against compiled implementation
		# requires compiling using cuda-radon-transform repository, available on Bitbucket
		t0 = time.time()
		check11 =  radon_projections_compiled_cuda(listofpatches, theta, circle_inscribed)
		t1 = time.time()
		check22 = radon_projections_skimage_python(listofpatches, theta, circle_inscribed)
		t2 = time.time()
		print("Radon projections time, compiled CUDA:  "+str(t1-t0)+" seconds")
		print("Radon projections time, python skimage: "+str(t2-t1)+" seconds")
		describe("check11", check11)
		describe("check22", check22)

		import cv2
		for ii in range(check11.shape[0]):
			checkdiff = np.fabs(check11[ii,:,:] - check22[ii,:,:])
			describe("checkdiff", checkdiff)
			zp = np.zeros((4,check11.shape[2]))
			concat = np.concatenate((check11[ii,:,:], zp, check22[ii,:,:], zp, checkdiff), axis=0)
			#cv2.imshow("npresult", uint8norm(concat))
			#cv2.waitKey(0)
	else:
		# run only one of the implementations
		radonfunc = radon_projections_skimage_python
		beftime = time.time()
		npresult = radonfunc(listofpatches, theta, circle_inscribed)
		#print("sinogram calculation took "+str(time.time()-beftime)+" seconds")
		#describe("python sinogram", npresult)
		assert len(npresult.shape) == 3, str(npresult.shape)

		# also do FFT + normalization as final stage of feature extraction
		# subtract 1 from normed which is the mean

		absproc = lambda xx: np.expand_dims(np.absolute(xx), axis=-1)
		beftime = time.time()
		_, fftnormed, _, fftavg = fftscores(npresult)

		npresult = absproc(fftnormed) - 1.
		#npresult = np.concatenate([absproc(fftnormed) - 1., absproc(fftavg)], axis=1)
		npresult=np.reshape(npresult,(1,18,46))
		print np.shape(npresult)
		#print("FFT calculations took "+str(time.time()-beftime)+" seconds")
		#describe("npresult", npresult)
	if q==0:
		resamp_feat=npresult			
		
	else:
		resamp_feat=np.concatenate((resamp_feat, npresult), axis=0)
	print np.shape(resamp_feat)
		
labels=np.zeros((np.shape(resamp_feat)[0],2))
labels[:,1]=1		
sio.savemat(destination+'radon_test_manip_image.mat',{'feat':resamp_feat,'labels':labels})

		
























