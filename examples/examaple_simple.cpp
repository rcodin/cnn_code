#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <layers.h>

void *alloc_2D(int i, int j, size_t bytes) {
	void *ret = malloc(i * j * bytes);

	return ret;
}

void *alloc_3D(int i, int j, int k, size_t bytes) {
	void *ret = malloc(i * j * k * bytes);

	return ret;
}

void *alloc_4D(int i, int j, int k, int l, size_t bytes) {
	void *ret = malloc(i * j * k * bytes);

	return ret;
}

void free_mem(void *ptr) {
	free(ptr);
}

int main()
{
	//conv->relu->pool conv->relu->pool conv->relu->pool fc->fc
	

	//load network weights
	size_t bytes = sizeof(float);
	float ***conv1_filter = (float ***)alloc_4D(3, 3, 3, 64, bytes);
	float ***conv2_filter = (float ***)alloc_4D(3, 3, 64, 128, bytes);
	float ***conv3_filter = (float ***)alloc_4D(3, 3, 128, 256, bytes);
	float **fc1 = (float **)alloc_2D(24*24*256, 256, bytes);
	float **fc2 = (float **)alloc_2D(256, 256, bytes);

	//create input 
	//conv1->relu->pool

	//Conv1
	struct Conv_conf conv1_conf;

	conv1_conf.h = 224;
	conv1_conf.w = 224
	conv1_conf.in_c = 3;
	conv1_conf.out_c = 64;
	conv1_conf.f_h = 3;
	conv1_conf.f_w = 3;

	float ***input = (float ***)alloc_3D(224, 224, 3);
	float ***output = (float ***)alloc_3D(74, 74, 64);
	conv_forward(input, out, conv1_filter, conv1_conf);

	//conv2->relu->pool

	//conv3->relu->pool


	//free memory
	free(conv1_filter);
	free(conv2_filter);
	free(conv3_filter);
	free(fc1);
	free(fc2);
}