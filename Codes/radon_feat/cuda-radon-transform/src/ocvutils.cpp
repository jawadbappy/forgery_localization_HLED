/*
Author: Jason Bunk
Jan. 2017
*/
#include "ocvutils.h"
#include <iostream>

#ifndef to_istring
template <class T>
inline std::string to_istring(const T& t)
{
	std::stringstream ss;
	ss << static_cast<int>(t);
	return ss.str();
}
template <class T>
inline std::string to_sstring(const T& t) {
	std::stringstream ss;
	ss << (t);
	return ss.str();
}
#endif
#define CVMAP_MAKETYPE(thecvtype) cvdepthsmap[thecvtype] = #thecvtype
static std::map<int,std::string> cvdepthsmap;
void init_cvdepthsmap() {
	CVMAP_MAKETYPE(CV_8U);
	CVMAP_MAKETYPE(CV_16U);
	CVMAP_MAKETYPE(CV_8S);
	CVMAP_MAKETYPE(CV_16S);
	CVMAP_MAKETYPE(CV_32F);
	CVMAP_MAKETYPE(CV_64F);
}
std::string describemat(const cv::Mat & inmat) {
	if(cvdepthsmap.empty()) {
		init_cvdepthsmap();
	}
	std::string retstr = std::string("(rows,cols,channels,depth) == (")
					+to_istring(inmat.rows)+std::string(", ")
					+to_istring(inmat.cols)+std::string(", ")
					+to_istring(inmat.channels())+std::string(", ");
	if(cvdepthsmap.find((int)inmat.depth()) != cvdepthsmap.end()) {
		retstr += cvdepthsmap[inmat.depth()]+std::string(")");
	} else {
		retstr += to_istring(inmat.depth())+std::string(")");
	}
	double minval,maxval;
	cv::minMaxIdx(inmat, &minval, &maxval);
	retstr += std::string(", (min,max) = (")+to_sstring(minval)+std::string(", ")+to_sstring(maxval)+std::string(")");
	return retstr;
}
