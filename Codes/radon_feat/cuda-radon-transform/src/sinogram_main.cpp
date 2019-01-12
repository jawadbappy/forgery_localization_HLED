/*
 * Adapted from this github link by Jason Bunk
 * https://github.com/rehmanali1994/easy_computed_tomography.github.io
 */
// Includes, system
#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

// OpenCV for reading/writing images
#include <opencv2/highgui/highgui.hpp>

// Include sinogram
#include "sinogram.h"

// misc
using std::cout; using std::endl;
#define pi acos(-1)


// Host code
int main(int argc, char *argv[])
{
	if(argc <= 3) {
		cout<<"usage:  {image-file}  {angles-file}  {circle_inscribed?}"<<endl;
		exit(1);
	}

	// Read image as grayscale
	cv::Mat inimg = cv::imread(argv[1], cv::IMREAD_GRAYSCALE);

	// Read Projection Angles from File
	float datfromfile;
	std::vector<float> h_angles;
	FILE *in_theta = fopen(argv[2], "r");
	if (in_theta == NULL) { fprintf(stderr, "Input file for theta info has some issues. Please check."); exit(1); }
	while(EOF != fscanf(in_theta, "%f", &datfromfile)) {
		h_angles.push_back(datfromfile * pi / 180.0f);
	}

	//
	int circle_inscribed = strtol(argv[3], NULL, 10);

	// compute transform
	cv::Mat returned = radon_transform(inimg, h_angles, circle_inscribed);
	cv::transpose(returned, returned);

	// save as normalized grayscale image
	double amin,amax;
	cv::minMaxIdx(returned, &amin, &amax);
	std::cout<<"output: (min,max) == ("<<amin<<", "<<amax<<")"<<std::endl;
	cv::imwrite("sinogram_out.png", (returned-amin)*(255.0/(amax-amin)));

	return 0;
}
