/*
 * Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 * Modifications: Jason Bunk, Jan. 2017
 *     convolutionNotSeparableGPU()
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "check_macros.h"

#include "convolutionTexture.h"

////////////////////////////////////////////////////////////////////////////////
// GPU-specific defines
////////////////////////////////////////////////////////////////////////////////
//Maps to a single instruction on G8x / G9x / G10x
//#define IMAD(a, b, c) ( __mul24((a), (b)) + (c) )
#define IMAD(a, b, c) ( ((a) * (b)) + (c) )

//Use unrolled innermost convolution loop
#define UNROLL_INNER 1

//Round a / b to nearest higher integer value
inline int iDivUp(int a, int b) {
    return (a % b != 0) ? (a / b + 1) : (a / b);
}

//Align a to nearest higher multiple of b
inline int iAlignUp(int a, int b) {
    return (a % b != 0) ? (a - a % b + b) : a;
}


////////////////////////////////////////////////////////////////////////////////
// Convolution kernel and input array storage
////////////////////////////////////////////////////////////////////////////////
__constant__ float c_Kernel[KERNEL_LENGTH];
__constant__ float c_notsep_Kernel[KERNEL_LENGTH*KERNEL_LENGTH];

void setConvolutionKernel(float *h_Kernel) {
    cudaMemcpyToSymbol(c_Kernel, h_Kernel, KERNEL_LENGTH * sizeof(float));
}
void setNotSeparableConvolutionKernel(float *h_Kernel) {
    cudaMemcpyToSymbol(c_notsep_Kernel, h_Kernel, KERNEL_LENGTH * KERNEL_LENGTH * sizeof(float));
}

//texture<float, 2, cudaReadModeElementType> texSrc;
// 2D float texture
texture<float, cudaTextureType2D, cudaReadModeElementType> texSrc;


void convolution_setTextureParams() {
    // Set texture reference parameters
    texSrc.addressMode[0] = cudaAddressModeClamp; // AddressModeBorder == zero outside the texture
    texSrc.addressMode[1] = cudaAddressModeClamp; // AddressModeClamp == last row or column is repeated
    texSrc.filterMode = cudaFilterModePoint; // disable interpolation (i.e. use "nearest-neighbor")
}

////////////////////////////////////////////////////////////////////////////////
// Loop unrolling templates, needed for best performance
////////////////////////////////////////////////////////////////////////////////
template<int i> __device__ float convolutionRow(float x, float y) {
    return tex2D(texSrc, x + (float)(KERNEL_RADIUS - i), y) * c_Kernel[i] + convolutionRow<i - 1>(x, y);
}

template<> __device__ float convolutionRow<-1>(float x, float y) {
    return 0;
}

template<int i> __device__ float convolutionColumn(float x, float y) {
    return tex2D(texSrc, x, y + (float)(KERNEL_RADIUS - i)) * c_Kernel[i] + convolutionColumn<i - 1>(x, y);
}

template<> __device__ float convolutionColumn<-1>(float x, float y) {
    return 0;
}


////////////////////////////////////////////////////////////////////////////////
// Row convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionRowsKernel(
    float *d_Dst,
    int imageW,
    int imageH
) {
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH) {
        return;
    }

    float sum = 0;

#if(UNROLL_INNER)
    sum = convolutionRow<2 *KERNEL_RADIUS>(x, y);
#else
    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
        sum += tex2D(texSrc, x + (float)k, y) * c_Kernel[KERNEL_RADIUS - k];
    }
#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}


void convolutionRowsGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
) {
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    CUDA_CHECK(cudaBindTextureToArray(texSrc, a_Src));
    convolutionRowsKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    CUDA_POST_KERNEL_CHECK;

    CUDA_CHECK(cudaUnbindTexture(texSrc));
}



////////////////////////////////////////////////////////////////////////////////
// Column convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionColumnsKernel(
    float *d_Dst,
    int imageW,
    int imageH
) {
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH) {
        return;
    }

    float sum = 0;

#if(UNROLL_INNER)
    sum = convolutionColumn<2 *KERNEL_RADIUS>(x, y);
#else
    for (int k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; k++) {
        sum += tex2D(texSrc, x, y + (float)k) * c_Kernel[KERNEL_RADIUS - k];
    }
#endif

    d_Dst[IMAD(iy, imageW, ix)] = sum;
}

void convolutionColumnsGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH
) {
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    CUDA_CHECK(cudaBindTextureToArray(texSrc, a_Src));
    convolutionColumnsKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH
    );
    CUDA_POST_KERNEL_CHECK;

    CUDA_CHECK(cudaUnbindTexture(texSrc));
}



////////////////////////////////////////////////////////////////////////////////
// Not-separable convolution filter
////////////////////////////////////////////////////////////////////////////////
__global__ void convolutionNotSeparableKernel(
    float *d_Dst,
    int imageW,
    int imageH,
    float normalization
) {
    const   int ix = IMAD(blockDim.x, blockIdx.x, threadIdx.x);
    const   int iy = IMAD(blockDim.y, blockIdx.y, threadIdx.y);
    const float  x = (float)ix + 0.5f;
    const float  y = (float)iy + 0.5f;

    if (ix >= imageW || iy >= imageH) {
        return;
    }

    float sum = 0;

    for(int ii = -KERNEL_RADIUS; ii <= KERNEL_RADIUS; ++ii) {
    for(int jj = -KERNEL_RADIUS; jj <= KERNEL_RADIUS; ++jj) {
        // technically cross-correlation so don't use with asymmetric kernels
        sum += tex2D(texSrc, x + static_cast<float>(ii), y + static_cast<float>(jj))
                    * c_notsep_Kernel[(ii+KERNEL_RADIUS) * KERNEL_LENGTH + (jj+KERNEL_RADIUS)];
    }
    }

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    d_Dst[IMAD(iy, imageW, ix)] += (sqrtf(fabs(sum)) / normalization);
}

void convolutionNotSeparableGPU(
    float *d_Dst,
    cudaArray *a_Src,
    int imageW,
    int imageH,
    float normalization
) {
    dim3 threads(16, 12);
    dim3 blocks(iDivUp(imageW, threads.x), iDivUp(imageH, threads.y));

    CUDA_CHECK(cudaBindTextureToArray(texSrc, a_Src));
    convolution_setTextureParams();
    convolutionNotSeparableKernel<<<blocks, threads>>>(
        d_Dst,
        imageW,
        imageH,
        normalization
    );
    CUDA_POST_KERNEL_CHECK;

    CUDA_CHECK(cudaUnbindTexture(texSrc));
}
