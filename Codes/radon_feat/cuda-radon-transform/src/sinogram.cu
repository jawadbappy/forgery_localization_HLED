/*
 * rotationKernel() from github link
 * https://github.com/rehmanali1994/easy_computed_tomography.github.io
 * adapted by Jason Bunk, Jan. 2017
 */

// Includes, system
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// Includes CUDA
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <cufft.h>

#include <opencv2/highgui/highgui.hpp>

// Our includes
#include "check_macros.h"
#include "sinogram.h"
#include "convolutionTexture.h"
#include "ocvutils.h"

// misc
using std::cout; using std::endl;
//#define DO_THE_FFT 1

//------------------------------------------
// From Caffe: CUDA: grid stride looping
#define CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
       i < (n); \
       i += blockDim.x * gridDim.x)
// CUDA: use 512 threads per block
const int CAFFE_CUDA_NUM_THREADS = 512;
// CUDA: number of blocks for threads.
inline int CAFFE_GET_BLOCKS(const int N) {
	return (N + CAFFE_CUDA_NUM_THREADS - 1) / CAFFE_CUDA_NUM_THREADS;
}
//------------------------------------------


// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texRef;


// Simple transformation kernel
__global__ void rotationKernel(float* d_sg, int numSensors, int numAngles, float widthX, float widthY, float const* theta)
{
	// Calculate texture coordinates
	const unsigned int r_idx = blockIdx.x * blockDim.x + threadIdx.x;
	const unsigned int t_idx = blockIdx.y * blockDim.y + threadIdx.y;

	if (r_idx < numSensors && t_idx < numAngles) {

		const float halfWidthR = 0.5f*static_cast<float>(numSensors);
		const float costheta = cosf(theta[t_idx]);
		const float sintheta = sinf(theta[t_idx]);
		const float r = static_cast<float>(r_idx) - halfWidthR;

		float integral = 0;

		for (int z_idx = 0; z_idx < numSensors; z_idx++) {

			float z = static_cast<float>(z_idx) - halfWidthR;

			// Transform coordinates
			float tr = (r * costheta + z * sintheta + 0.5f*widthX + 0.5f)/widthX;
			float tz = (z * costheta - r * sintheta + 0.5f*widthY + 0.5f)/widthY;

			// Read from texture
			integral += tex2D(texRef, tr, tz);
		}

		// write to global memory
		//d_sg[r_idx*numAngles + t_idx] = integral;
		d_sg[t_idx*numSensors + r_idx] = integral;
	}
}

__global__ void complexMagnitudeKernel(const int num, const cufftComplex *input, float* output) {
  CUDA_KERNEL_LOOP(index, num) {
    output[index] = cuCabsf(input[index]);
  }
}


//==============================================================================

void RadonTransformer::set_params(const std::vector<float> & ANGLES, bool CIRCLE_INSCRIBED, bool DIFF_FIRST) {
  CHECK_EQ(was_initialized_by_setting_params_, false);
	CHECK_EQ(was_initialized_for_fixed_size_, false);
  angles_ = ANGLES;
  circle_inscribed_ = CIRCLE_INSCRIBED;
  diff_first_ = DIFF_FIRST;
  was_initialized_by_setting_params_ = true;
}


void RadonTransformer::init_for_batch_of_fixed_size(int rows, int cols) {
	CHECK(was_initialized_by_setting_params_);
	CHECK_EQ(was_initialized_for_fixed_size_, false);
	free_arrays();
	was_initialized_for_fixed_size_ = true;

	// Get array sizes and coordinate bounds
	numX_ = cols;
	numY_ = rows;
	floatImgSize_ = numX_ * numY_ * sizeof(float);

	// more descriptors of input shapes
	const float widthX = static_cast<float>(numX_);
	const float widthY = static_cast<float>(numY_);
	const int numAngles = angles_.size();

	numSensors_ = circle_inscribed_ ? std::min(numX_,numY_) : static_cast<int>(ceilf(sqrtf(widthX*widthX + widthY*widthY)));
	numOutputs_ = numSensors_ * numAngles;

	// Set texture reference parameters
	texRef.addressMode[0] = cudaAddressModeBorder; // AddressModeBorder == zero outside the texture
	texRef.addressMode[1] = cudaAddressModeBorder; // AddressModeClamp == last row or column is repeated
	texRef.filterMode = cudaFilterModeLinear;
	texRef.normalized = true; // texture coordinates will be indexed in range [0, 1]

	// Allocate one large block of memory using one cudaMalloc() call
	mallocsize_ = 0;
	mallocsize_ += numAngles * sizeof(float);		// Allocate angles
	mallocsize_ += numOutputs_ * sizeof(float);	// Allocate result of transformation in device memory
	if(diff_first_) {
		mallocsize_ += floatImgSize_;	// Allocate result of diff
#if DO_THE_FFT
		mallocsize_ += numfftOutputs*sizeof(cufftComplex);	// Allocate result of fft
#endif
	}
	CUDA_CHECK(cudaMalloc(&bigmalloc_, mallocsize_));

	// Distribute pointers from bigmalloc accordingly
	uint8_t * mptr = bigmalloc_;
	d_angles_ = static_cast<float*>(static_cast<void*>(mptr));  mptr += numAngles * sizeof(float);
	d_sg_     = static_cast<float*>(static_cast<void*>(mptr));  mptr += numOutputs_ * sizeof(float);
#if DO_THE_FFT
	cufftComplex *myfft_dest = NULL;
#endif
	if(diff_first_) {
		diff_tmp_   =     static_cast<float*>(static_cast<void*>(mptr));  mptr += floatImgSize_;
#if DO_THE_FFT
		myfft_dest = static_cast<cufftComplex*>(static_cast<void*>(mptr));  mptr += numfftOutputs*sizeof(cufftComplex);
#endif
	}
	// Make sure we have distributed exactly the requested number of bytes
	CHECK_EQ(mptr - bigmalloc_, mallocsize_);

	// Allocate CUDA texture-like array in device memory
	channelDesc_ = new cudaChannelFormatDesc(cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat));
	CUDA_CHECK(cudaMallocArray(&cuArray_, channelDesc_, numX_, numY_));

	// Copy angles array to GPU. The very first part of the bigmalloc must be the angles.
	CUDA_CHECK(cudaMemcpy(static_cast<float*>(static_cast<void*>(bigmalloc_)),
						&angles_[0], numAngles * sizeof(float), cudaMemcpyHostToDevice));

  if(diff_first_) {
		// initialize convolution kernel to discrete 3x3 laplacian
		std::vector<float> convkernel(9, 0.0f);
		convkernel[1] = convkernel[3] = convkernel[5] = convkernel[7] = 1.0f;
		convkernel[4] = -4.0f;
		setNotSeparableConvolutionKernel(&convkernel[0]);
  }
}


void RadonTransformer::process_batch_of_fixed_size(std::vector<cv::Mat> & inputs) const {
	CHECK(was_initialized_by_setting_params_);
	CHECK(was_initialized_for_fixed_size_);
	CHECK(cuArray_ != NULL);

	// more descriptors of input shapes
	const float widthX = static_cast<float>(numX_);
	const float widthY = static_cast<float>(numY_);
	const int numAngles = angles_.size();
#if DO_THE_FFT
	const int numfftOutputs = (diff_first_ ? ((numSensors_/2)+1) : numSensors_) * numAngles;
	const int numfftSensors =  diff_first_ ? ((numSensors_/2)+1) : numSensors_;
#else
	const int numfftOutputs = numSensors_ * numAngles;
	const int numfftSensors = numSensors_;
#endif

	// get ready to loop
	const int num_inputs = inputs.size();

	// loop over batch
	for(int nn=0; nn<num_inputs; ++nn) {
		// convert image to a float32 array
		cv::Mat tmp32f;
		if(inputs[nn].depth() == CV_32F) {
			tmp32f = inputs[nn];
		} else {
			inputs[nn].convertTo(tmp32f, CV_32F);
		}
    CHECK_EQ(tmp32f.cols, numX_);
    CHECK_EQ(tmp32f.rows, numY_);
		// split color channels
		// if grayscale, will simply copy to first element of vector
		std::vector<cv::Mat> channels;
		cv::split(tmp32f, channels);
		CHECK_GE(channels.size(), 1);
		CHECK_EQ(channels.size(), inputs[nn].channels());
		CHECK_EQ(channels[0].rows, numY_); CHECK_EQ(channels[0].cols, numX_);

		if(!diff_first_) {
			CHECK_EQ(inputs[nn].channels(), 1);
			// Copy image to GPU memory, and bind the array to the texture reference
			CUDA_CHECK(cudaMemcpyToArray(cuArray_, 0, 0, static_cast<float*>(static_cast<void*>(tmp32f.data)),
																								floatImgSize_, cudaMemcpyHostToDevice));
			CUDA_CHECK(cudaBindTextureToArray(texRef, cuArray_, *channelDesc_));
		} else {
			// set diffs storage to zero
			CUDA_CHECK(cudaMemset(diff_tmp_, 0, floatImgSize_));
			// loop over each channel, sum the result in diff_tmp
			for(int chan=0; chan<channels.size(); ++chan) {
				// Copy image to GPU memory, and bind the array to the texture reference
				CUDA_CHECK(cudaMemcpyToArray(cuArray_, 0, 0, static_cast<float*>(static_cast<void*>(channels[chan].data)),
																									floatImgSize_, cudaMemcpyHostToDevice));
				CUDA_CHECK(cudaBindTextureToArray(texRef, cuArray_, *channelDesc_));
				convolutionNotSeparableGPU(diff_tmp_, cuArray_, numX_, numY_, static_cast<float>(channels.size()));
			}
			// copy from diff_tmp back to array, to prepare for radon kernel below
			CUDA_CHECK(cudaMemcpyToArray(cuArray_, 0, 0, diff_tmp_, floatImgSize_, cudaMemcpyDeviceToDevice));
	#if 0
			// test the convolution code by checking its output: uncomment this and see test_conv2d.py
			cv::Mat sepconvresult(numY_, numX_, CV_32F);
			CUDA_CHECK(cudaMemcpy(sepconvresult.data, diff_tmp_, floatImgSize_, cudaMemcpyDeviceToHost));
			//cout<<endl<<"sepconvresult: "<<endl<<describemat(sepconvresult)<<endl<<sepconvresult<<endl<<endl<<endl;
			double amin,amax; // save as normalized grayscale image
			cv::minMaxIdx(sepconvresult, &amin, &amax);
			std::cout<<"output: (min,max) == ("<<amin<<", "<<amax<<")"<<std::endl;
			cv::imwrite("sino_sepconvresult.png", (sepconvresult-amin)*(255.0/(amax-amin)));
	#endif
		}

		// Invoke Radon transformation kernel
		dim3 dimBlock(16, 16, 1);
		dim3 dimGrid((numSensors_ + dimBlock.x - 1) / dimBlock.x, (numAngles + dimBlock.y - 1) / dimBlock.y, 1);
		rotationKernel<<<dimGrid, dimBlock>>>(d_sg_, numSensors_, numAngles, widthX, widthY, d_angles_);
		CUDA_POST_KERNEL_CHECK;

#if 0
		// test the convolution code by checking its output: uncomment this and see test_conv2d.py
		cv::Mat radonresult(numAngles, numSensors_, CV_32F);
		CUDA_CHECK(cudaMemcpy(radonresult.data, d_sg_, numOutputs_*sizeof(float), cudaMemcpyDeviceToHost));
		double amin,amax; // save as normalized grayscale image
		cv::minMaxIdx(radonresult, &amin, &amax);
		std::cout<<"output: (min,max) == ("<<amin<<", "<<amax<<")"<<std::endl;
		cv::imwrite("sino_radonresult.png", (radonresult-amin)*(255.0/(amax-amin)));
#endif

#if DO_THE_FFT
		if(diff_first_) {
	    // CUFFT plan simple API
	    cufftHandle myPlan;
	    CUFFT_CHECK(cufftPlan1d(&myPlan, numSensors_, CUFFT_R2C, numAngles));
			// perform fft and get magnitude
			CUFFT_CHECK(cufftExecR2C(myPlan, d_sg_, myfft_dest));
			complexMagnitudeKernel<<<CAFFE_GET_BLOCKS(numfftOutputs),CAFFE_CUDA_NUM_THREADS>>>(numfftOutputs, myfft_dest, d_sg_);
			CUDA_POST_KERNEL_CHECK;
		}
#endif

		// Copy results from GPU to CPU
		inputs[nn] = cv::Mat(numAngles, numfftSensors, CV_32F);
		CUDA_CHECK(cudaMemcpy(static_cast<float*>(static_cast<void*>(inputs[nn].data)),
											d_sg_, numfftOutputs * sizeof(float), cudaMemcpyDeviceToHost));
	}
}


void RadonTransformer::free_arrays() {
	// Free memory
	if(channelDesc_ != NULL) {
		delete channelDesc_;
		channelDesc_ = NULL;
	}
	if(cuArray_ != NULL) {
		CUDA_CHECK(cudaFreeArray(cuArray_));
		cuArray_ = NULL;
	}
	if(bigmalloc_ != NULL) {
		CUDA_CHECK(cudaFree(bigmalloc_));
		bigmalloc_ = NULL;
	}
	was_initialized_for_fixed_size_ = false;
	numX_ = numY_ = numSensors_ = mallocsize_ = floatImgSize_ = numOutputs_ = 0;
}

void RadonTransformer::init_context() {
		// Initialize the driver API
		CUDA_CHECK(cuInit(0));
		// Get a handle to the last compute device
                int ndevices = -1;
                CUDA_CHECK(cuDeviceGetCount(&ndevices));
		CHECK_GT(ndevices, 0);
		CUDA_CHECK(cuDeviceGet(&cuCtx_device_, ndevices-1));
		// Create a compute device context
		CUDA_CHECK(cuCtxCreate(&cuCtx_context_, 0, cuCtx_device_));
}
void RadonTransformer::free_context() {
		CUDA_CHECK(cuCtxDestroy(cuCtx_context_));
}


// Compute radon transform, return sinogram as float32 cv::Mat
//       similar interface to Python scikit-image's radon()
cv::Mat radon_transform(const cv::Mat & input, const std::vector<float> & angles,
                          bool circle_inscribed, bool diff_first/*= true*/) {
	RadonTransformer transf;
	transf.set_params(angles, circle_inscribed, diff_first);
	transf.init_for_batch_of_fixed_size(input.rows, input.cols);
	std::vector<cv::Mat> inputs(1, input);
	transf.process_batch_of_fixed_size(inputs);
	return inputs[0];
}
