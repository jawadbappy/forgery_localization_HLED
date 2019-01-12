from random import randint
import numpy as np
import cv2
from utils import uint8_nonorm

def crop_to_bounding_box_from_alpha(img):
    assert len(img.shape) == 3 and img.shape[2] == 4, str(img.shape)
    assert img.dtype == np.uint8, str(img.dtype)
    # get contour of object from alpha channel, find bounding box around it, and crop
    smask = cv2.dilate(img[:,:,3], np.ones((3,3),dtype=np.uint8))
    contours,_ = cv2.findContours(smask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    assert len(contours) >= 1
    if len(contours) == 1:
        pts = contours[0]
    else:
        pts = np.concatenate(contours, axis=0)
    x,y,w,h = cv2.boundingRect(pts)
    return img[y:(y+h),x:(x+w),:]
    #return img[(y-1):(y+h+1),(x-1):(x+w+1),:]


def splice_crop_into_image(pim, crop):
    # pim == "pristine image"
    assert pim.dtype == np.uint8, str(pim.dtype)
    assert crop.dtype == np.uint8, str(crop.dtype)
    assert len(pim.shape) == 3 and len(crop.shape) == 3
    assert crop.shape[2] == 4, str(crop.shape)

    c_row, c_col = crop.shape[:2]
    p_row, p_col = pim.shape[:2]

    if c_row > p_row or c_col > p_col:
        print("crop.shape "+str(crop.shape)+", pim.shape "+str(pim.shape))
        print("skipped splicing \'"+sfile+"\' into \'"+pfile+"\'")
        return (None,None)

    assert (p_row-c_row-1) >= 0, str(p_row)+", "+str(c_row)
    assert (p_col-c_col-1) >= 0, str(p_col)+", "+str(c_col)

    c_y = randint(0, (p_row-c_row-1)) # randint() range is inclusive
    c_x = randint(0, (p_col-c_col-1))

    assert (p_row-c_y-c_row) >= 0
    assert (p_col-c_x-c_col) >= 0

    padded = np.pad(crop, ((c_y, p_row-c_y-c_row), (c_x, p_col-c_x-c_col), (0,0)), mode='constant')
    assert padded.shape[:2] == pim.shape[:2], "padded.shape == "+str(padded.shape)+", pim.shape == "+str(pim.shape)

    rgbfloat = pim.astype(np.float32) / 255.0
    padfloat = padded.astype(np.float32) / 255.0

    result_msk = padfloat[:,:,3].reshape(list(padfloat.shape[:2])+[1,])
    result_rgb = (rgbfloat[:,:,:3] * (1.0 - result_msk)) + (padfloat[:,:,:3] * result_msk)

    return uint8_nonorm(result_rgb), uint8_nonorm(result_msk)


def splice_color_transfer(source, target):
	"""
    http://www.pyimagesearch.com/2014/06/30/super-fast-color-transfer-images/
    https://github.com/jrosebr1/color_transfer

	Transfers the color distribution from the source to the target
	image using the mean and standard deviations of the L*a*b*
	color space.

	This implementation is (loosely) based on to the "Color Transfer
	between Images" paper by Reinhard et al., 2001.

	Parameters:
	-------
	source: NumPy array
		OpenCV image in BGR color space (the source image)
	target: NumPy array
		OpenCV image in BGR color space (the target image)

	Returns:
	-------
	transfer: NumPy array
		OpenCV image (w, h, 3) NumPy array (uint8)
	"""
	# convert the images from the RGB to L*ab* color space, being
	# sure to utilizing the floating point data type (note: OpenCV
	# expects floats to be 32-bit, so use that instead of 64-bit)
	source = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
	target = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
	# compute color statistics for the source and target images
	(lMeanSrc, lStdSrc, aMeanSrc, aStdSrc, bMeanSrc, bStdSrc) = Lab_image_stats(source)
	(lMeanTar, lStdTar, aMeanTar, aStdTar, bMeanTar, bStdTar) = Lab_image_stats(target)
	# subtract the means from the target image
	(l, a, b) = cv2.split(target)
	l -= lMeanTar
	a -= aMeanTar
	b -= bMeanTar
	# scale by the standard deviations
	l = (lStdTar / (lStdSrc + 1e-12)) * l
	a = (aStdTar / (aStdSrc + 1e-12)) * a
	b = (bStdTar / (bStdSrc + 1e-12)) * b
	# add in the source mean
	l += lMeanSrc
	a += aMeanSrc
	b += bMeanSrc
	# clip the pixel intensities to [0, 255] if they fall outside this range
	l = np.clip(l, 0, 255)
	a = np.clip(a, 0, 255)
	b = np.clip(b, 0, 255)
	# merge the channels together and convert back to the RGB color
	# space, being sure to utilize the 8-bit unsigned integer data type
	transfer = cv2.merge([l, a, b])
	transfer = cv2.cvtColor(transfer.astype(np.uint8), cv2.COLOR_LAB2BGR)
	# return the color transferred image
	return transfer

def Lab_image_stats(image):
	"""
	Parameters:
	-------
	image: NumPy array
		OpenCV image in L*a*b* color space

	Returns:
	-------
	Tuple of mean and standard deviations for the L*, a*, and b*
	channels, respectively
	"""
	# compute the mean and standard deviation of each channel
	(l, a, b) = cv2.split(image)
	(lMean, lStd) = (l.mean(), l.std())
	(aMean, aStd) = (a.mean(), a.std())
	(bMean, bStd) = (b.mean(), b.std())
	# return the color statistics
	return (lMean, lStd, aMean, aStd, bMean, bStd)
