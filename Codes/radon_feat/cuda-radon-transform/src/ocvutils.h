#ifndef _OPEN_CV_UTILS_H_____
#define _OPEN_CV_UTILS_H_____
/*
Author: Jason Bunk
Jan. 2017
*/

#include <string>
#include <opencv2/core/core.hpp>

// return a string describing the matrix (rows, columns, dtype, min/max, etc)
std::string describemat(const cv::Mat & inmat);

#endif
