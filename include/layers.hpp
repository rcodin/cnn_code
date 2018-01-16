#include <stdio.h>
#include <cstdlib>
#include <iostream>

struct Conv_conf {
	int h;
	int w;
};

struct Pool_conf {
	int h;
	int w;
};

struct Data_conf {
	int h;
	int w;
	int c;
};

enum Layer_type {
	conv,
	pool,
	relu,
	fc,
	no_imp
};

void conv_forward(float ***in, float ***out, float ****filter, Conv_conf conv_conf);
void pool_forward(float ***in, float ***out, Input_conf input_conf, Pool_conf pool_conf);
void relu_forward(float ***in, float ***out, Input_conf input_conf);
void linearize_conv(float ***in, float *out, float **filter, Input_conf input_conf, Input_conf output_conf);
void fc_forward(float *in, float *out, float **filter,int input_size, int output_size);