#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <layers.hpp>
#include <tiling.hpp>
#include <utils.hpp>
#include <mkl.h>

int main() {
	//224x224x3 Conv

	float *in = (float *)mkl_malloc(224*224*64 * sizeof(float), 32);
	float *out = (float *)mkl_malloc(224*224*3 * sizeof(float), 32);
	float *filter = (float *)mkl_malloc(64 * 3 * 3 * 3 * sizeof(float), 32);
	Conv_conf conv11_conf = {3, 3};
	
	Data_conf input11_conf = {224, 224, 64};
	Data_conf output11_conf = {224, 224, 64};

	conv_im2col(in, out, filter, conv11_conf, input11_conf, output11_conf);


	return 0;
}