#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <layers.hpp>

using namespace std;

void *alloc_2D(int i, int j, size_t bytes) {
	void **ret = (void **)malloc(i * sizeof(void *));

	for (int i_idx = 0; i_idx < i; i_idx++)
		ret[i_idx] = malloc(j * bytes);
	return ret;
}

void *alloc_3D(int i, int j, int k, size_t bytes) {
	void ***ret = (void ***)malloc(i * sizeof(void *));

	for (int i_idx = 0; i_idx < i; i_idx++) {
		ret[i_idx] = (void **)malloc(j * sizeof(void *));

		for (int j_idx = 0; j_idx < j; j_idx++)
			ret[i_idx][j_idx] = malloc(k * bytes);
	}

	// std::cout<<ret<<std::endl;
	if (ret == NULL)
		std::cout<<"Memory not allocated";
	return ret;
}

void *alloc_4D(int i, int j, int k, int l, size_t bytes) {
	void ****ret = (void ****)malloc(i * sizeof(void *));

	for (int i_idx = 0; i_idx < i; i_idx++) {
		ret[i_idx] = (void ***)malloc(j * sizeof(void *));

		for (int j_idx = 0; j_idx < j; j_idx++) {
			ret[i_idx][j_idx] = (void **)malloc(k * sizeof(void *));

			for (int k_idx = 0; k_idx < k; k_idx++)
				ret[i_idx][j_idx][k_idx] = malloc(l * bytes);
		}
	}

	// printf("%d\n", i * j * k * l * bytes);
	if (ret == NULL)
		std::cout<<"Memory not allocated";
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
	float ****conv1_filter = (float ****)alloc_4D(3, 3, 3, 64, bytes);

	// float conv1_filter[3][3][3][3];
	float ****conv2_filter = (float ****)alloc_4D(3, 3, 64, 128, bytes);
	float ****conv3_filter = (float ****)alloc_4D(3, 3, 128, 256, bytes);
	float **fc1_filter = (float **)alloc_2D(11*11*256, 512, bytes);
	float **fc2_filter = (float **)alloc_2D(512, 512, bytes);

	//create input 
	//conv1->relu->pool

	//Conv1


	Input_conf input11_conf;
	input_conf.h = 222;
	input_conf.w = 222;
	input_conf.c = 64;

	Input_conf input12_conf;

	Input_conf input13_conf;
	struct Conv_conf conv1_conf;

	conv1_conf.h = 224;
	conv1_conf.w = 224;
	conv1_conf.in_c = 3;
	conv1_conf.out_c = 64;
	conv1_conf.f_h = 3;
	conv1_conf.f_w = 3;


	struct Conv_conf conv2_conf;

	conv2_conf.h = 74;
	conv2_conf.w = 74;
	conv2_conf.in_c = 64;
	conv2_conf.out_c = 128;
	conv2_conf.f_h = 3;
	conv2_conf.f_w = 3;

	struct Conv_conf conv3_conf;

	conv3_conf.h = 24;
	conv3_conf.w = 24;
	conv3_conf.in_c = 128;
	conv3_conf.out_c = 256;
	conv3_conf.f_h = 3;
	conv3_conf.f_w = 3;
	
	float ***input = (float ***)alloc_3D(224, 224, 3, bytes);
	// float input[60][60][3];
	float ***output = (float ***)alloc_3D(222, 222, 64, bytes);
	// float output[58][58][3];
	// for (int i = 0; i < 100000000; i++)

	conv_forward(input, output, conv1_filter, conv1_conf);

	relu_forward(output, output, input_conf);

	free_mem(input);
	// free_mem(output);
	input = output;
	output = (float ***)alloc_3D(74, 74, 64, bytes);

	Pool_conf pool1_conf;
	
	pool1_conf.h = 3;
	pool1_conf.w = 3;
	
	pool_forward(input, output, input_conf, pool1_conf);


	//conv2->relu->pool



	free_mem(input);
	input = output;
	output = (float ***)alloc_3D(72, 72, 128, bytes);
	conv_forward(input, output, conv2_filter, conv2_conf);

	input_conf.h = 72;
	input_conf.w = 72;
	input_conf.c = 128;
	relu_forward(output, output, input_conf);

	free_mem(input);
	input = output;
	output = (float ***)alloc_3D(24, 24, 128, bytes);

	Pool_conf pool2_conf;
	
	pool2_conf.h = 3;
	pool2_conf.w = 3;
	pool_forward(input, output, input_conf, pool2_conf);


	// pool_forward();
	

	//conv3->relu->pool


	free_mem(input);
	input = output;
	output = (float ***)alloc_3D(24, 24, 256, bytes);
	conv_forward(input, output, conv3_filter, conv3_conf);

	input_conf.h = 22;
	input_conf.w = 22;
	input_conf.c = 256;
	relu_forward(output, output, input_conf);

	free_mem(input);
	input = output;
	output = (float ***)alloc_3D(11, 11, 256, bytes);

	Pool_conf pool3_conf;
	
	pool3_conf.h = 2;
	pool3_conf.w = 2;
	pool_forward(input, output, input_conf, pool3_conf);


	//fc1
	Input_conf output_conf;
	free_mem(input);
	input = output;
	float *fc_output = (float *)malloc(512 * bytes);

	input_conf.h = 11;
	input_conf.w = 11;
	input_conf.c = 256;
	output_conf.h = 512;
	linearize_conv(input, fc_output, fc1_filter, input_conf, output_conf);


	//fc2
	int input_size = 512;
	int output_size = 512;
	free_mem(input);
	float *fc_input;

	fc_input = fc_output;
	fc_output = (float *)malloc(output_size);
	fc_forward(fc_input, fc_output, fc2_filter, input_size, output_size);
	//free memory
	// free(conv1_filter);
	// free(conv2_filter);
	// free(conv3_filter);
	// free(fc1);
	// free(fc2);
}