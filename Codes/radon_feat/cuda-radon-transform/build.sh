#!/bin/bash

MY_NUMPY_INCLUDE=$(dirname `python -c 'import numpy.core; print(numpy.core.__file__)'`)/include

echo $MYPYINCLUDE

(mkdir -p build && cd build && \
	cmake .. \
	-DNUMPY_INCLUDE_PATH=$MY_NUMPY_INCLUDE \
	-DPYTHON_INCLUDE_DIR=$(python -c "from distutils.sysconfig import get_python_inc; print(get_python_inc())") \
	-DPYTHON_LIBRARY=$(python -c "import distutils.sysconfig as sysconfig; print(sysconfig.get_config_var('LIBDIR'))") \
	&& make)

