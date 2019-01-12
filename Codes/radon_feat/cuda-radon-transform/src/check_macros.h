#ifndef CHECK_MACROS_HPP_
#define CHECK_MACROS_HPP_
/*
Author: Jason Bunk
Jan. 2017
	This file is a simplified emulation of the Google Logging Library (glog),
		but this requires no compiliation or installation.
	Simply include this header in your project.
*/

#include <iostream>
static std::ostream unprinted_output_stops_here(NULL);
#define DONTPRINT unprinted_output_stops_here




#ifndef LOG
#ifndef INFO
#ifndef ERROR
#ifndef FATAL
enum log_info_type {
	INFO,
	ERROR,
	FATAL
};
#define DEFAULTERRMSG "\n@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@" \
		<<std::endl<<std::flush<< __FILE__ << ":" << __LINE__ <<" "

#define LOG(x) x == INFO ? std::cout<<"INFO: " : \
	(x == ERROR ? std::cout<<DEFAULTERRMSG "ERROR: " : \
	(x == FATAL ? std::cout<<DEFAULTERRMSG "FATAL-ERROR: " : \
	std::cout << #x ))

#endif
#endif
#endif
#endif




#ifndef CHECK
#ifndef CHECK_EQ
#ifndef CHECK_GT

#define BUILDCHKMSG(a,b,op) #a << op << #b << "  VALUES:  " << (a) << op << (b) <<std::endl<<std::flush

#define DEFAULTCHKMSG "CHECK FAILED:  "
#define CHECK(a)          (a)    ? DONTPRINT : LOG(ERROR) << DEFAULTCHKMSG << #a <<std::endl<<std::flush
#define CHECK_EQ(a,b) ((a)==(b)) ? DONTPRINT : LOG(ERROR) << DEFAULTCHKMSG << BUILDCHKMSG(a,b," == ")
#define CHECK_GT(a,b) ((a)> (b)) ? DONTPRINT : LOG(ERROR) << DEFAULTCHKMSG << BUILDCHKMSG(a,b," > " )
#define CHECK_GE(a,b) ((a)>=(b)) ? DONTPRINT : LOG(ERROR) << DEFAULTCHKMSG << BUILDCHKMSG(a,b," >= ")
#define CHECK_LT(a,b) ((a)< (b)) ? DONTPRINT : LOG(ERROR) << DEFAULTCHKMSG << BUILDCHKMSG(a,b," < " )
#define CHECK_LE(a,b) ((a)<=(b)) ? DONTPRINT : LOG(ERROR) << DEFAULTCHKMSG << BUILDCHKMSG(a,b," <= ")

#endif
#endif
#endif

#include "cuda_driver_errors.h"

#ifndef CUDA_CHECK__base
#define CUDA_CHECK__base(call,descfunc) \
    { int result = (call); if(result != CUDA_SUCCESS) { \
        cudaError_t err = cudaGetLastError(); \
        std::cerr << "CUDA error calling " #call ", code is " << err << ", result was " << descfunc(result) << std::endl; \
        exit(-1); } }
#endif



#ifndef CUDA_CHECK
#define CUDA_CHECK(call) CUDA_CHECK__base(call, _cudaGetErrorEnum)
#endif

#ifdef _CUFFT_H_
#define CUFFT_CHECK(call) CUDA_CHECK__base(call, _cufftGetErrorEnum)
#endif

#define CUDA_POST_KERNEL_CHECK CUDA_CHECK(cudaPeekAtLastError())


#endif
