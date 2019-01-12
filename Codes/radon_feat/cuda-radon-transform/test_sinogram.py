"""
Author: Jason Bunk

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
thispath = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(thispath,'build')) # for importing pysinogram.so
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import time
import skimage.io
from skimage.transform import radon
from utils import *

imagefname = 'SheppLoganPhantom.png'

circle_inscribed = False

# Load Image and Give Pixels (x,y) Coordinates
orig_object = skimage.io.imread(os.path.join('images',imagefname), flatten=True) # flatten means grayscale

print("## loaded image \'"+str(imagefname)+"\'")

# Projection Angles for Computed Tomography
numAngles = 360
theta = np.linspace(0,180,numAngles,endpoint=False)

# Save Important Values to .txt files before running CUDA
np.savetxt('theta.txt', theta)

# Run the CUDA executable interface to create the sinogram
exestr = os.path.join(thispath,'build','sinogram_main')+' '+os.path.join('images',imagefname)+' theta.txt '+' '+str(int(circle_inscribed))
beftime = time.time()
if os.system(exestr) == 0:
	print("CUDA executable interface: sinogram calculation took "+str(time.time()-beftime)+" seconds")
else:
	print("error in CUDA executable?")

# Run the python interface to the CUDA to create the sinogram
from pysinogram import BatchRadonTransform
beftime = time.time()
# transpose to match the format of skimage.transform.radon, which has an angle per column
npresult = BatchRadonTransform([orig_object,], list(theta), circle_inscribed)[0].transpose()
print("CUDA python interface: sinogram calculation took "+str(time.time()-beftime)+" seconds")
describe("python sinogram", npresult)
skimage.io.imsave('sinogram_out_pythoninterface.png', uint8norm(npresult.copy()))

# get ready for the scikit
scikitsinofile = 'sinogram_out_scikit.png'
scikitinput = orig_object
if True:
	kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]]) # 2D discrete 3x3 laplacian
	scikitinput = np.sqrt(np.fabs(signal.convolve2d(orig_object, kernel, mode='same', boundary='symm')))

	# transpose again to re-match the output format of BatchRadonTransform, which has an angle per row
	plotfftscores(scikitsinofile[:-4]+'_recreated_from_np', fftscores(npresult.transpose()))

# Run the much slower scikit-image radon() implementation
beftime = time.time()
scikitsinogram = radon(scikitinput, theta=theta, circle=circle_inscribed)
print("Scikit-image sinogram calculation took "+str(time.time()-beftime)+" seconds")
skimage.io.imsave(scikitsinofile, uint8norm(scikitsinogram.copy()))
# transpose to match the output format of BatchRadonTransform, which has an angle per row
plotfftscores(scikitsinofile[:-4], fftscores(scikitsinogram.transpose()))

plt.show()
