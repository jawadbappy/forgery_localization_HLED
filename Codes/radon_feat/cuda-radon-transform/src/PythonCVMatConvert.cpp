# include "PythonCVMatConvert.h"
#include <iostream>
using std::cout; using std::endl;
/*
 * https://github.com/yati-sagade/opencv-ndarray-conversion
 * The following conversion functions are taken/adapted from OpenCV's cv2.cpp file
 * inside modules/python/src2 folder.
 */

static void init()
{
    import_array();
}

static int failmsg(const char *fmt, ...)
{
    char str[1000];

    va_list ap;
    va_start(ap, fmt);
    vsnprintf(str, sizeof(str), fmt, ap);
    va_end(ap);

    PyErr_SetString(PyExc_TypeError, str);
    return 0;
}

class PyAllowThreads
{
public:
    PyAllowThreads() : _state(PyEval_SaveThread()) {}
    ~PyAllowThreads()
    {
        PyEval_RestoreThread(_state);
    }
private:
    PyThreadState* _state;
};

class PyEnsureGIL
{
public:
    PyEnsureGIL() : _state(PyGILState_Ensure()) {}
    ~PyEnsureGIL()
    {
        PyGILState_Release(_state);
    }
private:
    PyGILState_STATE _state;
};

using namespace cv;

static PyObject* failmsgp(const char *fmt, ...)
{
  char str[1000];

  va_list ap;
  va_start(ap, fmt);
  vsnprintf(str, sizeof(str), fmt, ap);
  va_end(ap);

  PyErr_SetString(PyExc_TypeError, str);
  return 0;
}

class MyNumpyAllocator : public MatAllocator
{
public:
    MyNumpyAllocator() {}
    ~MyNumpyAllocator() {}

    void allocate(int dims, const int* sizes, int type, int*& refcount,
                  uchar*& datastart, uchar*& data, size_t* step)
    {
        PyEnsureGIL gil;

        int depth = CV_MAT_DEPTH(type);
        int cn = CV_MAT_CN(type);
        const int f = (int)(sizeof(size_t)/8);
        int typenum = depth == CV_8U ? NPY_UBYTE : depth == CV_8S ? NPY_BYTE :
                      depth == CV_16U ? NPY_USHORT : depth == CV_16S ? NPY_SHORT :
                      depth == CV_32S ? NPY_INT : depth == CV_32F ? NPY_FLOAT :
                      depth == CV_64F ? NPY_DOUBLE : f*NPY_ULONGLONG + (f^1)*NPY_UINT;
        int i;
        npy_intp _sizes[CV_MAX_DIM+1];
        for( i = 0; i < dims; i++ )
        {
            _sizes[i] = sizes[i];
        }

        if( cn > 1 )
        {
            _sizes[dims++] = cn;
        }

        PyObject* o = PyArray_SimpleNew(dims, _sizes, typenum);

        if(!o)
        {
            CV_Error_(CV_StsError, ("The numpy array of typenum=%d, ndims=%d can not be created", typenum, dims));
        }
        refcount = refcountFromPyObject(o);

        npy_intp* _strides = PyArray_STRIDES(o);
        for( i = 0; i < dims - (cn > 1); i++ )
            step[i] = (size_t)_strides[i];
        datastart = data = (uchar*)PyArray_DATA(o);
    }

    void deallocate(int* refcount, uchar*, uchar*)
    {
        PyEnsureGIL gil;
        if( !refcount )
            return;
        PyObject* o = pyObjectFromRefcount(refcount);
        Py_INCREF(o);
        Py_DECREF(o);
    }
};

MyNumpyAllocator g_numpyAllocator;

NDArrayConverter::NDArrayConverter() { init(); }

void NDArrayConverter::init()
{
    import_array();
}

cv::Mat NDArrayConverter::toMat(const PyObject *o)
{
    cv::Mat m;

    if(!o || o == Py_None) {
        if( !m.data )
            m.allocator = &g_numpyAllocator;
    }

    if( !PyArray_Check(o) ) {
        failmsg("toMat: Object is not a numpy array");
    }

    int typenum = PyArray_TYPE(o);
    int type = typenum == NPY_UBYTE ? CV_8U : typenum == NPY_BYTE ? CV_8S :
               typenum == NPY_USHORT ? CV_16U : typenum == NPY_SHORT ? CV_16S :
               typenum == NPY_INT || typenum == NPY_LONG ? CV_32S :
               typenum == NPY_FLOAT ? CV_32F :
               typenum == NPY_DOUBLE ? CV_64F : -1;

    if( type < 0 ) {
        failmsg("toMat: Data type = %d is not supported", typenum);
    }

    int ndims = PyArray_NDIM(o);

    if(ndims >= CV_MAX_DIM) {
        failmsg("toMat: Dimensionality (=%d) is too high", ndims);
    }

    int size[CV_MAX_DIM+1];
    size_t step[CV_MAX_DIM+1], elemsize = CV_ELEM_SIZE1(type);
    const npy_intp* _sizes = PyArray_DIMS(o);
    const npy_intp* _strides = PyArray_STRIDES(o);
#if 0
    bool transposed = false;

    for(int i = 0; i < ndims; i++) {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }
    for(int i = 0; i < ndims; i++) {
      cout<<"before: ndims == "<<ndims<<", size["<<i<<"] == "<<size[i]<<", step == "<<step[i]<<endl;
    }
    if( ndims == 0 || step[ndims-1] > elemsize ) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }
    if( ndims >= 2 && step[0] < step[1] ) {
        std::swap(size[0], size[1]);
        std::swap(step[0], step[1]);
        transposed = true;
    }
    if( ndims == 3 && size[2] <= CV_CN_MAX && step[1] == elemsize*size[2] ) {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }
    if( ndims > 2) { failmsg("toMat: Object has more than 2 dimensions"); }

    for(int i = 0; i < ndims; i++) {
      cout<<"after: ndims == "<<ndims<<", size["<<i<<"] == "<<size[i]<<", step == "<<step[i]<<endl;
    }
    m = Mat(ndims, size, type, PyArray_DATA(o), step);

    if( m.data ) {
        m.refcount = refcountFromPyObject(o);
        m.addref(); // protect the original numpy array from deallocation
                    // (since Mat destructor will decrement the reference counter)
    };
    m.allocator = &g_numpyAllocator;

    if( transposed ) {
        Mat tmp;
        tmp.allocator = &g_numpyAllocator;
        transpose(m, tmp);
        m = tmp;
    }
#else
    PyArrayObject* oarr = (PyArrayObject*) o;
    bool ismultichannel = ndims == 3 && _sizes[2] <= CV_CN_MAX;
    bool needcopy = false;

    for( int i = ndims-1; i >= 0 && !needcopy; i-- ) {
        // these checks handle cases of
        //  a) multi-dimensional (ndims > 2) arrays, as well as simpler 1- and 2-dimensional cases
        //  b) transposed arrays, where _strides[] elements go in non-descending order
        //  c) flipped arrays, where some of _strides[] elements are negative
        if( (i == ndims-1 && (size_t)_strides[i] != elemsize) ||
            (i < ndims-1 && _strides[i] < _strides[i+1]) )
            needcopy = true;
    }
    if( ismultichannel && _strides[1] != (npy_intp)elemsize*_sizes[2] ) { needcopy = true; }

    if (needcopy) {
        //if (info.outputarg) {
        //    failmsg("Layout of the output array %s is incompatible with cv::Mat (step[ndims-1] != elemsize or step[1] != elemsize*nchannels)", info.name);
        //    return false;
        //}
        oarr = PyArray_GETCONTIGUOUS(oarr);
        o = (PyObject*) oarr;
        _strides = PyArray_STRIDES(oarr);
    }
    for(int i = 0; i < ndims; i++) {
        size[i] = (int)_sizes[i];
        step[i] = (size_t)_strides[i];
    }
    //for(int i = 0; i < ndims; i++) {
    //  cout<<"before: ndims == "<<ndims<<", size["<<i<<"] == "<<size[i]<<", step == "<<step[i]<<endl;
    //}
    // handle degenerate case
    if( ndims == 0) {
        size[ndims] = 1;
        step[ndims] = elemsize;
        ndims++;
    }
    if( ismultichannel ) {
        ndims--;
        type |= CV_MAKETYPE(0, size[2]);
    }
    if( ndims > 2 ) {
        failmsg("toMat: has more than 2 dimensions");
        cout<<"toMat: has more than 2 dimensions"<<endl;
    }

    //for(int i = 0; i < ndims; i++) {
    //  cout<<"after: ndims == "<<ndims<<", size["<<i<<"] == "<<size[i]<<", step == "<<step[i]<<endl;
    //}
    m = Mat(ndims, size, type, PyArray_DATA(oarr), step);

    if( m.data ) {
        m.refcount = refcountFromPyObject(o);
        if (!needcopy) {
            m.addref(); // protect the original numpy array from deallocation
                        // (since Mat destructor will decrement the reference counter)
        }
    }
    m.allocator = &g_numpyAllocator;
#endif
    //cout<<"finished: m.rows == "<<m.rows<<", m.cols == "<<m.cols<<", m.channels() == "<<m.channels()<<endl;
    return m;
}

PyObject* NDArrayConverter::toNDArray(const cv::Mat& m)
{
    if( !m.data )
        Py_RETURN_NONE;
    Mat temp, *p = (Mat*)&m;
    if(!p->refcount || p->allocator != &g_numpyAllocator)
    {
        temp.allocator = &g_numpyAllocator;
        m.copyTo(temp);
        p = &temp;
    }
    p->addref();
    return pyObjectFromRefcount(p->refcount);
}

//=================================================================================
// Misc utils

void AddPathToPythonSys(std::string path) {
	// now insert the current working directory into the python path so module search can take advantage
	// this must happen after python has been initialised

	//cout << "adding to python path: \"" << path << "\"" << endl;
        char pathchar[] = "path";
	PyObject* sysPath = PySys_GetObject(pathchar);
	PyList_Insert(sysPath, 0, PyString_FromString(path.c_str()));
	//print python's search paths to confirm that it was added
	/*PyRun_SimpleString(	"import sys\n"
				"from pprint import pprint\n"
				"pprint(sys.path)\n");*/
}

static bool GLOBAL_PYTHON_WAS_INITIALIZED = false;
/*#include <mutex>
static std::mutex GLOBAL_PYTHON_INITIALIZING_MUTEX;*/
static PyThreadState * state = NULL;

bool PrepareForPythonInterpreter() {
	//GLOBAL_PYTHON_INITIALIZING_MUTEX.lock();
	if(GLOBAL_PYTHON_WAS_INITIALIZED == false) {
		if(Py_IsInitialized()) {
			cout<<"WARNING -- PrepareForPythonInterpreter() -- PYTHON INTERPETER ALREADY INITIALIZED?????"<<endl;
		}
		Py_Initialize();
		PyEval_InitThreads();
		state = PyEval_SaveThread();
		if(!Py_IsInitialized()) {
			cout<<"WARNING -- PrepareForPythonInterpreter() -- FAILED TO INITIALIZE PYTHON INTERPETER"<<endl;
		}
		GLOBAL_PYTHON_WAS_INITIALIZED = true;
	}
	//GLOBAL_PYTHON_INITIALIZING_MUTEX.unlock();
	return (Py_IsInitialized() != 0);
}
