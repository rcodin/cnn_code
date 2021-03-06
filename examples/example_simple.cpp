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
	Conv_conf conv1_conf = {3, 3};
	
	Data_conf input11_conf = {224, 224, 3};
	Data_conf output11_conf = {222, 222, 64};

	//relu1
	Data_conf input12_conf = {222, 222, 64};
	Data_conf output12_conf = {222, 222, 64};

	//pool1
	Data_conf input13_conf = {222, 222, 64};
	Data_conf output13_conf = {74, 74, 64};
	Pool_conf pool1_conf = {3, 3};
	
	//Conv2
	struct Conv_conf conv2_conf = {3, 3};
	Data_conf input21_conf = {74, 74, 64};
	Data_conf output21_conf = {72, 72, 128};

	//relu2
	Data_conf input22_conf = {72, 72, 128};
	Data_conf output22_conf = {72, 72, 128};

	//pool2
	Data_conf input23_conf = {72, 72, 128};	
	Data_conf output23_conf = {24, 24, 128};
	Pool_conf pool2_conf = {3, 3};

	struct Conv_conf conv3_conf = {3, 3};
	//Conv3	
	Data_conf input31_conf = {24, 24, 128};
	Data_conf output31_conf = {22, 22, 256};

	//relu3
	Data_conf input32_conf = {22, 22, 256};
	Data_conf output32_conf = {22, 22, 256};


	//pool3
	Data_conf input33_conf = {22, 22, 256};
	Data_conf output33_conf = {11, 11, 256};
	Pool_conf pool3_conf = {2, 2};


	//conv1->relu->pool
	float ***input = (float ***)alloc_3D(input11_conf.h, input11_conf.h, input11_conf.c, bytes);
	float ***output = (float ***)alloc_3D(output11_conf.h, output11_conf.h, output11_conf.c, bytes);
	conv_forward(input, output, conv1_filter, conv1_conf, input11_conf, output11_conf);
	relu_forward(output, output, input12_conf);
	free_mem(input);
	input = output;
	output = (float ***)alloc_3D(output13_conf.h, output13_conf.w, output13_conf.c, bytes);
	pool_forward(input, output, input12_conf, pool1_conf);
	

	//conv2->relu->pool
	free_mem(input);
	input = output;
	output = (float ***)alloc_3D(output21_conf.h, output21_conf.w, output21_conf.c, bytes);
	conv_forward(input, output, conv2_filter, conv2_conf, input21_conf, output21_conf);
	relu_forward(output, output, input22_conf);
	free_mem(input);
	input = output;
	output = (float ***)alloc_3D(output23_conf.h, output23_conf.w, output23_conf.c, bytes);
	pool_forward(input, output, input23_conf, pool2_conf);
	
	//conv3->relu->pool
	free_mem(input);
	input = output;
	output = (float ***)alloc_3D(output31_conf.h, output31_conf.w, output31_conf.c, bytes);
	conv_forward(input, output, conv3_filter, conv3_conf, input31_conf, output31_conf);
	relu_forward(output, output, input32_conf);
	free_mem(input);
	input = output;
	output = (float ***)alloc_3D(output33_conf.h, output33_conf.w, output33_conf.c, bytes);
	pool_forward(input, output, input33_conf, pool3_conf);

	//fc1
	Data_conf input_conf;
	Data_conf output_conf;
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