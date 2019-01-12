/*
Author: Jason Bunk
Jan. 2017
*/
#include <iostream>
#include <stdint.h>
#include <boost/python.hpp>
#include <thread>
#include "PythonCVMatConvert.h"
#include "utils_RNG.h"

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace bp = boost::python;
using std::cout; using std::endl;

#define DEBUG_NO_THREADING 1

#define MAX_N_THREADS 6

//static RNG_rand_r myrng; //no longer used due to parallelization (each thread needs its own RNG)

template <class T>
inline std::string to_istring(const T& t)
{
	std::stringstream ss;
	ss << static_cast<int>(t);
	return ss.str();
}
std::string describemat(const cv::Mat & inmat) {
	return std::string("(rows,cols,channels,depth) == (")
		+to_istring(inmat.rows)+std::string(", ")
		+to_istring(inmat.cols)+std::string(", ")
		+to_istring(inmat.channels())+std::string(", ")
		+to_istring(inmat.depth())
		+std::string(")");
}

/*
scX*cos(rth) + shB*sin(rth),
shA*cos(rth) + scY*sin(rth),
-(scX*cos(rth) + shB*sin(rth))*cX - (shA*cos(rth) + scY*sin(rth))*cY + cX + trX,

shB*cos(rth) - scX*sin(rth),
scY*cos(rth) - shA*sin(rth),
-(shB*cos(rth) - scX*sin(rth))*cX - (scY*cos(rth) - shA*sin(rth))*cY + cY + trY,
*/

#define ROTATION_BASE_AMT  20.0f
#define SHEAR_BASE_AMT      3.5f
#define TRANSLATE_BASE_AMT 0.03f

void processbatch_randcrop(RNG* myRNG, std::vector<cv::Mat>* batch, int extraarg, cv::Mat * optionalMat)
{
	// extraarg == "crop width" and "crop height" (returns square crops)
	const int numimgs = ((int)batch->size());
	int cropR, cropC;
	int marginR, marginC;

	if(extraarg <= 0) {
		std::cout<<"Crop size "<<extraarg<<" must be > 0 !!!!!!!!!"<<std::endl<<std::flush;
		return;
	}

	for(int ii=0; ii<numimgs; ii++) {
		marginR = ((*batch)[ii].rows - extraarg);
		marginC = ((*batch)[ii].cols - extraarg);
		if(marginR >= 0 && marginC >= 0) {
			cropR = myRNG->rand_int(0, marginR);
			cropC = myRNG->rand_int(0, marginC);
			(*batch)[ii] = (*batch)[ii](cv::Rect(cropC,cropR,extraarg,extraarg)).clone();
		} else {
			std::cout<<"IMAGE WAS SMALLER THAN CROP SIZE: "<<(*batch)[ii].cols<<"x"<<(*batch)[ii].rows<<" > "<<extraarg<<"x"<<extraarg<<std::endl<<std::flush;
		}
	}
}

void processbatch_randaffinetransf(RNG* myRNG, std::vector<cv::Mat>* batch, int extraarg, cv::Mat * optionalMat)
{
	//extraarg has three bits:
	// 0 i.e. 0001: "do scaling?" if false, does rotations, shears, and translations only
	// 1 i.e. 0010: "flip?" if true, will randomly mirror across X axis
	// 2 i.e. 0100: "fill border with 0s?" if true, will fill border with 0s, else will use REFLECT_101
	// 3 i.e. 1000: translations?
	const int numimgs = ((int)batch->size());
	float cX, cY, trX, trY, rth, shA, shB, scX, scY;
	cv::Mat transfmat(2, 3, CV_32F);

	for(int ii=0; ii<numimgs; ii++) {

		cX = ((float)(*batch)[ii].cols) * 0.5f;
		cY = ((float)(*batch)[ii].rows) * 0.5f;

		if(optionalMat == NULL) {
			//translation
			if(extraarg & 8) {
				cout<<"RANDOM TRANSLATIONS!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
				trX = myRNG->rand_float(-TRANSLATE_BASE_AMT*cX, TRANSLATE_BASE_AMT*cX);
				trY = myRNG->rand_float(-TRANSLATE_BASE_AMT*cY, TRANSLATE_BASE_AMT*cY);
			} else {
				trX = trY = 0.0f;
			}
			//rotation; negative angles make it look more italicized: (-6.8, 8.0) approximately centers
			rth = 0.0f;
			while(fabs(rth) < 1.0f) {
				rth = myRNG->rand_float(-ROTATION_BASE_AMT, ROTATION_BASE_AMT); // degrees to radians: multiply by pi/180
			}
			rth *= 0.01745329252f;

			//shear
			shA = myRNG->rand_float(-SHEAR_BASE_AMT, SHEAR_BASE_AMT)/cX; //negative A makes it look more italicized: (-3.0,3.7) approximately centers
			shB = myRNG->rand_float(-SHEAR_BASE_AMT, SHEAR_BASE_AMT)/cY; //positive B makes it look more italicized: (-3.7,3.0) approximately centers

			//scaling?
			if(extraarg & 1) {
				scX = 1.0f; scY = 1.0f;
				while(fabs(scX-1.0f) < 0.02f && fabs(scY-1.0f) < 0.02f) {
					scX = myRNG->rand_float(0.66f, 1.5f); //if scaled slightly too big, that's OK
					scY = scX * myRNG->rand_float(0.95f, 1.0f/0.95f); //different scaling between X and Y
				}
			} else {
				cout<<"NO SCALING!!!!!!!!!!!!!!!!!!!!!!!!"<<endl;
				scX = myRNG->rand_float(0.96f, 1.0f/0.96f);
				scY = myRNG->rand_float(0.96f, 1.0f/0.96f);
			}

			// now create 3 affine transformation matrices, one for each warp type (translate, rotate, shear), and then combine them
			// order: translate to centered about origin; rotate; rescale; shear; translate back and offset

			transfmat.at<float>(0,0) = scX*cos(rth) + shB*sin(rth);
			transfmat.at<float>(0,1) = shA*cos(rth) + scY*sin(rth);
			transfmat.at<float>(0,2) = -(scX*cos(rth) + shB*sin(rth))*cX - (shA*cos(rth) + scY*sin(rth))*cY + cX + trX;

			transfmat.at<float>(1,0) = shB*cos(rth) - scX*sin(rth);
			transfmat.at<float>(1,1) = scY*cos(rth) - shA*sin(rth);
			transfmat.at<float>(1,2) = -(shB*cos(rth) - scX*sin(rth))*cX - (scY*cos(rth) - shA*sin(rth))*cY + cY + trY;
			cout<<"USING AUTO-GENERATED TRANSFORMATION!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"<<endl<<std::flush;
		} else {
			transfmat = (*optionalMat);
		}

		//transfmat.at<float>(0,0) = ; transfmat.at<float>(0,1) = ; transfmat.at<float>(0,2) = ;
		//transfmat.at<float>(1,0) = ; transfmat.at<float>(1,1) = ; transfmat.at<float>(1,2) = ;

		//cout<<"tmat "<<transfmat<<endl<<std::flush;
		//cout<<"s"<<ii<<" to "<<(*batch)[ii].size().width<<"x"<<(*batch)[ii].size().height<<endl<<std::flush;

		cv::Mat temp;
		if((extraarg & 2) && myRNG->rand_float(0.0f, 1.0f) < 0.5f) {
			cv::flip((*batch)[ii], temp, 1);
		} else {
			(*batch)[ii].copyTo(temp);
		}

		//cout<<"s"<<ii<<" to "<<temp.size().width<<"x"<<temp.size().height<<endl<<std::flush;
		//cv::imshow("input", (*batch)[ii] / 255.0f);
		//cv::imshow("temp", temp / 255.0f);
		//cv::waitKey(0);

		cv::warpAffine(temp, (*batch)[ii], transfmat, (*batch)[ii].size(), cv::INTER_LINEAR,
						(extraarg & 4) ? cv::BORDER_CONSTANT : cv::BORDER_REFLECT_101);

		//cout<<"f"<<ii<<endl<<std::flush;
	}
}

void processbatch_jpeg(RNG* myRNG, std::vector<cv::Mat>* batch, int extraarg, cv::Mat * optionalMat)
{
	const int numimgs = ((int)batch->size());
	std::vector<int> params(3,0);
	params[0] = CV_IMWRITE_JPEG_QUALITY;
	std::vector<uint8_t> tmp;
	for(int ii=0; ii<numimgs; ii++) {
		if(extraarg > 0) {
			params[1] = extraarg; // JPEG quality factor
		} else {
			params[1] = myRNG->rand_int(60, 97);
		}
		cv::imencode(std::string(".jpg"), (*batch)[ii], tmp, params);
		(*batch)[ii] = cv::imdecode(tmp, cv::IMREAD_ANYCOLOR);
	}
}

void processbatch_affinetransf_thencrop(RNG* myRNG, std::vector<cv::Mat>* batch, int extraarg, cv::Mat * optionalMat)
{
	// extraarg:
	// 1: "do scaling?" if extraarg > 1,000,000 then will do scaling, else will not
	// 2: "crop width" and "crop height" (returns square crops)
	// to do both scaling and specify crop width (e.g. to 28), feed 1,000,028 == 1000028

	int cropsize = (extraarg & 1048575); // bits 0-19 are reserved for the crop size
	int transfargs = ((((uint32_t)extraarg) & 15728640) >> 20) & 15; // bits 20,21,22,23 are for the transformation args

	processbatch_randaffinetransf(myRNG, batch, transfargs, optionalMat);
	processbatch_randcrop(myRNG, batch, cropsize, NULL);
}


bp::object OnlineBatchProcessGeneric(bp::list* pyImagesBatch,
																		void (*processorFuncPtr)(RNG*, std::vector<cv::Mat>*, int, cv::Mat*),
																		int extraarg,
																		PyObject *optionalPythonMat = NULL)
{
	int ii, jj;
	const int numimgs = bp::len(*pyImagesBatch);

	if(numimgs <= 0) {
		cout<<"OnlineBatchRandomAffineTransf: warning: no images in batch"<<endl<<std::flush;
		return bp::object();
	}

	NDArrayConverter cvt;

	cv::Mat optionalMat;
	bool got_optional_mat = false;
	if(optionalPythonMat != NULL) {
		if(optionalPythonMat != Py_None) {
			optionalMat = cvt.toMat(optionalPythonMat);
			got_optional_mat = (!optionalMat.empty() && optionalMat.rows > 0 && optionalMat.cols > 0);
		}
	}

	// imgsperthread is ceil(numimgs / numthreads)
	const int numthreads = numimgs < MAX_N_THREADS ? numimgs : MAX_N_THREADS;
	const int imgsperthread = 1 + ((numimgs - 1) / numthreads);

	std::vector< std::vector<cv::Mat>* > vecImagesBatches(numthreads); // outer vector indexes threads, inner vector indexes images per thread
	std::vector<std::thread*> threads(numthreads);
	std::vector<RNG_rand_r*> threadRNGs(numthreads); // RNG for each thread
	bp::list returnedPyImages; //will be a list of cv2 numpy images

	int idx = -1;
	for(ii=0; ii < numthreads; ii++) {
		vecImagesBatches[ii] = new std::vector<cv::Mat>();
		for(jj=0; jj < imgsperthread; jj++) {
			idx++;
			if(idx < numimgs) {
				bp::object imgobject = bp::extract<bp::object>((*pyImagesBatch)[idx]);
				vecImagesBatches[ii]->push_back(cvt.toMat(imgobject.ptr()));
				if((*vecImagesBatches[ii])[jj].depth() != CV_32F) {
					(*vecImagesBatches[ii])[jj].convertTo((*vecImagesBatches[ii])[jj], CV_MAKETYPE(CV_32F, (*vecImagesBatches[ii])[jj].channels()));
				}
			}
		}
		//cout<<"launching thread with "<<vecImagesBatches[ii]->size()<<" images; numthreads "<<numthreads<<", imgsperthread "<<imgsperthread<<endl<<std::flush;

		threadRNGs[ii] = new RNG_rand_r(rand());//ii);
#if DEBUG_NO_THREADING
		//cout<<"debug: no threading"<<endl<<std::flush;
		(*processorFuncPtr)(threadRNGs[ii], vecImagesBatches[ii], extraarg,
																	got_optional_mat ? &optionalMat : NULL);
#else
		threads[ii] = new std::thread(processorFuncPtr, threadRNGs[ii], vecImagesBatches[ii], extraarg,
																	got_optional_mat ? &optionalMat : NULL);
		//cout<<"waiting for thread "<<(ii+1)<<" / "<<numthreads<<" to join"<<endl<<std::flush;
		//threads[ii]->join();
#endif
	}

	for(ii=0; ii < numthreads; ii++) {
#if DEBUG_NO_THREADING
#else
		cout<<"waiting for thread "<<(ii+1)<<" / "<<numthreads<<" to join"<<endl<<std::flush;
		threads[ii]->join();
#endif
		idx = (int)vecImagesBatches[ii]->size();
		for(jj=0; jj<idx; jj++) {
			PyObject* thisImgCPP = cvt.toNDArray((*vecImagesBatches[ii])[jj]);
			//returnedPyImages.append(bp::object(bp::handle<>(bp::borrowed(thisImgCPP)))); // borrowed() increments the reference counter, causing a memory leak when we return to Python
			returnedPyImages.append(bp::object(bp::handle<>(thisImgCPP)));
		}
		delete vecImagesBatches[ii];
		delete threadRNGs[ii];
#if DEBUG_NO_THREADING
#else
		delete threads[ii];
#endif
	}

	return returnedPyImages;
}

bp::object OnlineBatchRandomAffineTransf(bp::list pyImagesBatch, int doresize, PyObject *optionalTransfMat = NULL) {
	return OnlineBatchProcessGeneric(&pyImagesBatch, &processbatch_randaffinetransf, doresize, optionalTransfMat);
}
bp::object OnlineBatchRandomCrop(bp::list pyImagesBatch, int cropsize) {
	return OnlineBatchProcessGeneric(&pyImagesBatch, &processbatch_randcrop, cropsize);
}
bp::object OnlineBatchRandomAffineThenCrop(bp::list pyImagesBatch, int extraarg, PyObject *optionalTransfMat = NULL) {
	return OnlineBatchProcessGeneric(&pyImagesBatch, &processbatch_affinetransf_thencrop, extraarg, optionalTransfMat);
}
bp::object OnlineBatchJPEG(bp::list pyImagesBatch, int extraarg) {
	return OnlineBatchProcessGeneric(&pyImagesBatch, &processbatch_jpeg, extraarg);
}

static void init()
{
	Py_Initialize();
	import_array();
}

BOOST_PYTHON_MODULE(pymyonlineaffinecpplib)
{
	init();
	bp::def("OnlineBatchRandomAffineTransf", OnlineBatchRandomAffineTransf);
	bp::def("OnlineBatchRandomCrop", OnlineBatchRandomCrop);
	bp::def("OnlineBatchRandomAffineThenCrop", OnlineBatchRandomAffineThenCrop);
	bp::def("OnlineBatchJPEG", OnlineBatchJPEG);
}
