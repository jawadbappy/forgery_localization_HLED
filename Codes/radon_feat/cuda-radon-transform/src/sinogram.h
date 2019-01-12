#ifndef __CU_SINOGRAM_H______
#define __CU_SINOGRAM_H______
/*
 * Author: Jason Bunk
 * Jan. 2017
 */

#include <opencv2/core/core.hpp>
#include <vector>
#include <cuda_runtime.h>
#include "cuda_driver_errors.h"


class RadonTransformer
{
public:
  void set_params(const std::vector<float> & ANGLES, bool CIRCLE_INSCRIBED, bool DIFF_FIRST);
  void init_for_batch_of_fixed_size(int rows, int cols);
  void process_batch_of_fixed_size(std::vector<cv::Mat> & inputs) const;

  // initializer
  RadonTransformer() : numX_(0), numY_(0), numSensors_(0), mallocsize_(0), floatImgSize_(0), numOutputs_(0),
                      was_initialized_by_setting_params_(false), was_initialized_for_fixed_size_(false),
                      channelDesc_(NULL), d_angles_(NULL), d_sg_(NULL), diff_tmp_(NULL), cuArray_(NULL), bigmalloc_(NULL)
                      {init_context();}
  ~RadonTransformer() {free_arrays(); free_context();}
protected:
  void init_context();
  void free_context();
  void free_arrays();

  std::vector<float> angles_;
  int numX_, numY_, numSensors_, mallocsize_, floatImgSize_, numOutputs_;
  bool circle_inscribed_;
  bool diff_first_;
  bool was_initialized_by_setting_params_, was_initialized_for_fixed_size_;
  cudaChannelFormatDesc * channelDesc_;
  float *d_angles_, *d_sg_, *diff_tmp_;
  cudaArray * cuArray_;
  uint8_t * bigmalloc_;

  CUdevice cuCtx_device_;
  CUcontext cuCtx_context_;
};


// Compute radon transform, return sinogram as float32 cv::Mat
//       similar interface to Python scikit-image's radon()
cv::Mat radon_transform(const cv::Mat & input, const std::vector<float> & angles,
                          bool circle_inscribed, bool diff_first = true);

#endif
