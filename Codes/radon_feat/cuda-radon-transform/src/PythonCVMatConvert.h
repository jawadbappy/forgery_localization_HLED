# ifndef __PYTHON_CVMAT_COVNERSION_H__
# define __PYTHON_CVMAT_COVNERSION_H__
/*
 * https://github.com/yati-sagade/opencv-ndarray-conversion
 */

#include <Python.h>
#include <opencv2/core/core.hpp>
#include "numpy/ndarrayobject.h"

static PyObject* opencv_error = 0;

static int failmsg(const char *fmt, ...);

class PyAllowThreads;

class PyEnsureGIL;

#define ERRWRAP2(expr) \
try \
{ \
    PyAllowThreads allowThreads; \
    expr; \
} \
catch (const cv::Exception &e) \
{ \
    PyErr_SetString(opencv_error, e.what()); \
    return 0; \
}

static PyObject* failmsgp(const char *fmt, ...);

static size_t REFCOUNT_OFFSET = (size_t)&(((PyObject*)0)->ob_refcnt) +
    (0x12345678 != *(const size_t*)"\x78\x56\x34\x12\0\0\0\0\0")*sizeof(int);

static inline PyObject* pyObjectFromRefcount(const int* refcount)
{
    return (PyObject*)((size_t)refcount - REFCOUNT_OFFSET);
}

static inline int* refcountFromPyObject(const PyObject* obj)
{
    return (int*)((size_t)obj + REFCOUNT_OFFSET);
}

enum { ARG_NONE = 0, ARG_MAT = 1, ARG_SCALAR = 2 };

class NDArrayConverter
{
private:
    void init();
public:
    NDArrayConverter();
    cv::Mat toMat(const PyObject* o);
    PyObject* toNDArray(const cv::Mat& mat);
};

//=================================================================================
// Misc utils

#include <boost/python.hpp>

void AddPathToPythonSys(std::string path);

template<class T>
boost::python::list std_vector_to_py_list(const std::vector<T>& v) {
	boost::python::list l;
	typename std::vector<T>::const_iterator it;
	for (it = v.begin(); it != v.end(); ++it)
		l.append(*it);
	return l;
}

template<class T>
std::vector<T> py_list_to_std_vector(const boost::python::list& list) {
	std::vector<T> vec;
	for(int ii=0; ii < boost::python::len(list); ++ii) {
		vec.push_back(boost::python::extract<T>(list[ii]));
	}
	return vec;
}

bool PrepareForPythonInterpreter();

// Use when C++ might try to launch multiple Python interpreters
struct aquire_py_GIL {
	PyGILState_STATE state;
	aquire_py_GIL() {
		state = PyGILState_Ensure();
	}
	~aquire_py_GIL() {
		PyGILState_Release(state);
	}
};

// Use this when a Python script needs to call some external C++ utility; put it in the C++ utility function
struct release_py_GIL {
	PyThreadState *state;
	release_py_GIL() {
		state = PyEval_SaveThread();
	}
	~release_py_GIL() {
		PyEval_RestoreThread(state);
	}
};

# endif
