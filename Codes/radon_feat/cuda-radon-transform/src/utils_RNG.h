#ifndef __UTILS_RANDOM_NG_HPP__
#define __UTILS_RANDOM_NG_HPP__

/**
 * Thread-safe RNG classes
 * Author: Jason Bunk
**/

#include <stdlib.h>
#include <ctime>


class RNG
{
public:
	virtual void Seed_And_Initialize() = 0;
	virtual int rand_int() = 0;
	virtual void Assign_Seed(unsigned int newseed) {}

	//--------

	// random integer: [min,max] inclusive of max
	virtual int rand_int(int min_value, int max_value) {
		return (max_value-min_value) <= 0 ? 0 : min_value + (rand_int() % (max_value - min_value + 1));
	}
	virtual float rand_float(float min_value, float max_value) {
		return min_value + ((max_value-min_value) * (((float)rand_int()) / ((float)RAND_MAX)));
	}
	virtual double rand_double(double min_value, double max_value) {
		return min_value + ((max_value-min_value) * (((double)rand_int()) / ((double)RAND_MAX)));
	}

	static double rand_double_static(double min_value, double max_value) {
		return min_value + ((max_value-min_value) * (((double)rand()) / ((double)RAND_MAX)));
	}
};


class RNG_rand_r : public RNG
{
protected:
	unsigned int myseed;
public:

	RNG_rand_r() {
		Seed_And_Initialize();
	}
	RNG_rand_r(unsigned int seedoffset) {
		myseed = time(nullptr) + seedoffset;
	}

	virtual void Seed_And_Initialize() {
		myseed = time(nullptr);
	}
	virtual void Assign_Seed(unsigned int newseed) {
		myseed = newseed;
	}
	virtual int rand_int() {
		return rand_r(&myseed);
	}
};

#endif
