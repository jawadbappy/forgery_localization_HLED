cmake_minimum_required(VERSION 2.8)
project(sinogramcuda)

set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR})
set(LIBRARY_OUTPUT_PATH ${CMAKE_BINARY_DIR})


# if on MAC then use clang++ with libc++
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
set(CMAKE_CXX_COMPILER "clang++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Darwin")
# if on linux then use g++
if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
set(CMAKE_CXX_COMPILER "g++")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -fPIC")
endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")

# Cuda flags
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -O2 -Xcompiler -fPIC" )


#find_package(rt REQUIRED)
include(CheckLibraryExists)
check_library_exists(rt clock_gettime "time.h" HAVE_CLOCK_GETTIME)

find_package(OpenCV REQUIRED)
find_package(PythonLibs 2.7 REQUIRED)
find_package(CUDA REQUIRED)


add_subdirectory(src)
