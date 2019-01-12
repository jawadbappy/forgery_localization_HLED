set(cusrcfiles
	sinogram.cu
	convolutionTexture.cu
	)

set(utilsfiles
	ocvutils.cpp
	)
set(pyutilsfiles
	PythonCVMatConvert.cpp
	)

set(pylibfiles
	boost_python_interface.cpp
	)

cuda_add_library(sinogram ${cusrcfiles})
target_link_libraries(sinogram rt ${OpenCV_LIBS} utilslib cuda)

add_library(utilslib ${utilsfiles})
add_library(pyutilslib ${pyutilsfiles})

add_executable(sinogram_main sinogram_main.cpp)
target_link_libraries(sinogram_main sinogram)


add_library(pysinogram MODULE ${pylibfiles})
set_target_properties(pysinogram PROPERTIES PREFIX "")
target_link_libraries(pysinogram sinogram pyutilslib python2.7 boost_python-py27)
include_directories(${CMAKE_SOURCE_DIR} ${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_PATH})



set(onlineaffinesrcfiles
	OnlineAffineTransf.cpp
	)
set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
add_library(pymyonlineaffinecpplib MODULE ${onlineaffinesrcfiles})
set_target_properties(pymyonlineaffinecpplib PROPERTIES PREFIX "")
target_link_libraries(pymyonlineaffinecpplib pyutilslib python2.7 boost_python-py27 ${OpenCV_LIBS})
include_directories(${PYTHON_INCLUDE_PATH} ${NUMPY_INCLUDE_PATH})
