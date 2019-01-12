/*
Author: Jason Bunk
Jan. 2017
*/
#include <iostream>
#include <stdint.h>
#include "PythonCVMatConvert.h"
#include <opencv2/core/core.hpp>
#include "sinogram.h"

namespace bp = boost::python;
using std::cout; using std::endl;


bp::object BatchRadonTransform(bp::list pyImagesBatch, bp::list thetas, int circle_inscribed, int do_laplacian_first)
{
	const int numimgs = bp::len(pyImagesBatch);
	const int numthetas = bp::len(thetas);
	if(numimgs <= 0) {
		cout<<"error: no images in batch"<<endl<<std::flush;
		return bp::object();
	}
	if(numthetas <= 0) {
		cout<<"error: numthetas <= 0"<<endl<<std::flush;
		return bp::object();
	}
	std::vector<float> thetas_cpp = py_list_to_std_vector<float>(thetas);
	for(int ii=0; ii<thetas_cpp.size(); ++ii) {
		thetas_cpp[ii] *= 0.01745329252f; // degrees to radians
	}

	NDArrayConverter cvt;
	bp::list returnedPyImages; //will be a list of cv2 numpy images
	std::vector<cv::Mat> tmp(numimgs);

	for(int ii=0; ii < numimgs; ii++) {
		bp::object imgobject = bp::extract<bp::object>(pyImagesBatch[ii]);
		tmp[ii] = cvt.toMat(imgobject.ptr());
	}

	// scope contains destruction of RadonTransformer and its CUDA context
	{
		RadonTransformer transf;
		transf.set_params(thetas_cpp, circle_inscribed, do_laplacian_first);
		transf.init_for_batch_of_fixed_size(tmp[0].rows, tmp[0].cols);
		transf.process_batch_of_fixed_size(tmp);
	}

	for(int ii=0; ii < numimgs; ii++) {
		PyObject* thisImgCPP = cvt.toNDArray(tmp[ii]);
		//returnedPyImages.append(bp::object(bp::handle<>(bp::borrowed(thisImgCPP)))); // borrowed() increments the reference counter, causing a memory leak when we return to Python
		returnedPyImages.append(bp::object(bp::handle<>(thisImgCPP)));
	}

	return returnedPyImages;
}


static void init() {
	Py_Initialize();
	import_array();
}
BOOST_PYTHON_MODULE(pysinogram)
{
	init();
	bp::def("BatchRadonTransform", BatchRadonTransform);
}
