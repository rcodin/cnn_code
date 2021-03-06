/*
Neural Network with 2 tiled and fused layers
*/

#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <layers.hpp>
#include <tiling.hpp>

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
	float ****conv1_filter = (float ****)alloc_4D(64, 3, 3, 3, bytes);

	// float conv1_filter[3][3][3][3];
	float ****conv2_filter = (float ****)alloc_4D(128, 3, 3, 64, bytes);
	float ****conv3_filter = (float ****)alloc_4D(256, 3, 3, 128, bytes);
	float **fc1_filter = (float **)alloc_2D(11*11*256, 512, bytes);
	float **fc2_filter = (float **)alloc_2D(512, 512, bytes);

	//create input 
	//conv1->relu->pool

	//Conv1
	Conv_conf conv1_conf = {3, 3};
	
	Data_conf input11_conf = {990, 990, 3};
	Data_conf output11_conf = {990, 990, 64};

	//relu1
	Data_conf input12_conf = {990, 990, 64};
	Data_conf output12_conf = {990, 990, 64};

	//pool1
	Data_conf input13_conf = {330, 330, 64};
	Data_conf output13_conf = {330, 330, 64};
	Pool_conf pool1_conf = {3, 3};
	
	//Conv2
	struct Conv_conf conv2_conf = {3, 3};
	Data_conf input21_conf = {332, 332, 64};
	Data_conf output21_conf = {330, 330, 128};

	//relu2
	Data_conf input22_conf = {330, 330, 128};
	Data_conf output22_conf = {330, 330, 128};

	//pool2
	Data_conf input23_conf = {330, 330, 128};	
	Data_conf output23_conf = {110, 110, 128};
	Pool_conf pool2_conf = {3, 3};

	struct Conv_conf conv3_conf = {3, 3};
	//Conv3	
	Data_conf input31_conf = {110, 110, 128};
	Data_conf output31_conf = {108, 108, 256};

	//relu3
	Data_conf input32_conf = {108, 108, 256};
	Data_conf output32_conf  = {108, 108, 256};


	//pool3
	Data_conf input33_conf = {108, 108, 256};
	Data_conf output33_conf = {36, 36, 256};
	Pool_conf pool3_conf = {3, 3};

	int h_num_tiles = 10;
	int w_num_tiles = 10;

	Data_conf output23_tiled_conf = {output23_conf.h/h_num_tiles, output23_conf.w/w_num_tiles, output23_conf.c};
	Data_conf input23_tiled_conf = {output23_tiled_conf.h * pool2_conf.h, output23_tiled_conf.w * pool2_conf.w, output23_conf.c};
	
	Data_conf output22_tiled_conf = {input23_tiled_conf.h, input23_tiled_conf.w, output22_conf.c};
	Data_conf input22_tiled_conf = {output22_tiled_conf.h, output22_tiled_conf.w, input22_conf.c};

	Data_conf output21_tiled_conf = {input22_tiled_conf.h, input22_tiled_conf.w, output21_conf.c};
	Data_conf input21_tiled_conf = {output21_tiled_conf.h + (conv2_conf.h - 1), output21_tiled_conf.w + (conv2_conf.w - 1), input21_conf.c};

	Data_conf output13_tiled_conf = {input21_tiled_conf.h, input21_tiled_conf.w, output13_conf.c};
	Data_conf input13_tiled_conf = {output13_tiled_conf.h * pool1_conf.h, output13_tiled_conf.w * pool1_conf.w, input13_conf.c};

	Data_conf output12_tiled_conf = {input13_tiled_conf.h, input13_tiled_conf.w, output12_conf.c};
	Data_conf input12_tiled_conf = {output12_tiled_conf.h, output12_tiled_conf.w, input12_conf.c};

	Data_conf output11_tiled_conf = {input12_tiled_conf.h, input12_tiled_conf.w, output11_conf.c};
	Data_conf input11_tiled_conf = {output11_tiled_conf.h + (conv1_conf.h - 1), output11_tiled_conf.w + (conv1_conf.w - 1), input11_conf.c};

	tile_idx_conf input11_tile_mult, input12_tile_mult, input13_tile_mult, input21_tile_mult, input22_tile_mult, input23_tile_mult;
	tile_idx_conf output11_tile_mult, output12_tile_mult, output13_tile_mult, output21_tile_mult, output22_tile_mult, output23_tile_mult;
	
	//pool2
	output23_tile_mult = {output23_tiled_conf.h, output23_tiled_conf.w, output23_tiled_conf.c};
	input23_tile_mult = {output23_tile_mult.h_base_idx * pool2_conf.h, output23_tile_mult.w_base_idx * pool2_conf.w, output23_tile_mult.c_base_idx};

	output22_tile_mult = {input23_tile_mult.h_base_idx, input23_tile_mult.w_base_idx, input23_tile_mult.c_base_idx};
	input22_tile_mult = {output22_tile_mult.h_base_idx, output22_tile_mult.w_base_idx, output22_tile_mult.c_base_idx};

	output21_tile_mult = {input22_tile_mult.h_base_idx, input22_tile_mult.w_base_idx, input22_tile_mult.c_base_idx};
	input21_tile_mult = {output21_tile_mult.h_base_idx, output21_tile_mult.w_base_idx, input21_tiled_conf.c};

	output13_tile_mult = {input21_tile_mult.h_base_idx, input21_tile_mult.w_base_idx, input21_tile_mult.c_base_idx};			
	input13_tile_mult = {output13_tile_mult.h_base_idx * pool2_conf.h, output13_tile_mult.w_base_idx * pool2_conf.w, output13_tile_mult.c_base_idx};

	output12_tile_mult = {input13_tile_mult.h_base_idx, input13_tile_mult.w_base_idx, input13_tile_mult.c_base_idx};
	input12_tile_mult = {output12_tile_mult.h_base_idx, output12_tile_mult.w_base_idx, output12_tile_mult.c_base_idx};

	output11_tile_mult = {input12_tile_mult.h_base_idx, input12_tile_mult.w_base_idx, input12_tile_mult.c_base_idx};
	input11_tile_mult = {output11_tile_mult.h_base_idx, output11_tile_mult.w_base_idx, input11_tiled_conf.c};


	tile_idx_conf input11_tile_base, input12_tile_base, input13_tile_base, input21_tile_base, input22_tile_base, input23_tile_base;
	tile_idx_conf output11_tile_base, output12_tile_base, output13_tile_base, output21_tile_base, output22_tile_base, output23_tile_base;

	float ***input11 = (float ***)alloc_3D(input11_tiled_conf.h, input11_tiled_conf.w, input11_tiled_conf.c, bytes);
	float ***output11 = (float ***)alloc_3D(output11_tiled_conf.h, output11_tiled_conf.w, output11_tiled_conf.c, bytes);
	
	float ***output12 = (float ***)alloc_3D(output12_tiled_conf.h, output12_tiled_conf.w, output12_tiled_conf.c, bytes);
	float ***output13 = (float ***)alloc_3D(output13_tiled_conf.h, output13_tiled_conf.w, output13_tiled_conf.c, bytes);
	float ***output21 = (float ***)alloc_3D(output21_tiled_conf.h, output21_tiled_conf.w, output21_tiled_conf.c, bytes);
	float ***output22 = (float ***)alloc_3D(output22_tiled_conf.h, output22_tiled_conf.w, output22_tiled_conf.c, bytes);
	float ***output23 = (float ***)alloc_3D(output23_tiled_conf.h, output23_tiled_conf.w, output23_tiled_conf.c, bytes);

	#pragma omp parallel for collapse(2)
	for (int h_tilem = 0; h_tilem < h_num_tiles; h_tilem++) {
		for (int w_tilem = 0; w_tilem < h_num_tiles; w_tilem++) {
			//conv1->relu->pool
			int h_tile = 0;
			int w_tile = 0;
			input11_tile_base = {input11_tile_mult.h_base_idx * h_tile, input11_tile_mult.w_base_idx * w_tile, input11_tile_mult.c_base_idx};
			input12_tile_base = {input12_tile_mult.h_base_idx * h_tile, input12_tile_mult.w_base_idx * w_tile, input12_tile_mult.c_base_idx};
			input13_tile_base = {input13_tile_mult.h_base_idx * h_tile, input13_tile_mult.w_base_idx * w_tile, input13_tile_mult.c_base_idx};
			input21_tile_base = {input21_tile_mult.h_base_idx * h_tile, input21_tile_mult.w_base_idx * w_tile, input21_tile_mult.c_base_idx};
			input22_tile_base = {input22_tile_mult.h_base_idx * h_tile, input22_tile_mult.w_base_idx * w_tile, input22_tile_mult.c_base_idx};
			input23_tile_base = {input23_tile_mult.h_base_idx * h_tile, input23_tile_mult.w_base_idx * w_tile, input23_tile_mult.c_base_idx};

			output11_tile_base = {output11_tile_mult.h_base_idx * h_tile, output11_tile_mult.w_base_idx * w_tile, output11_tile_mult.c_base_idx};
			output12_tile_base = {output12_tile_mult.h_base_idx * h_tile, output12_tile_mult.w_base_idx * w_tile, output12_tile_mult.c_base_idx};
			output13_tile_base = {output13_tile_mult.h_base_idx * h_tile, output13_tile_mult.w_base_idx * w_tile, output13_tile_mult.c_base_idx};
			output21_tile_base = {output21_tile_mult.h_base_idx * h_tile, output21_tile_mult.w_base_idx * w_tile, output21_tile_mult.c_base_idx};
			output22_tile_base = {output22_tile_mult.h_base_idx * h_tile, output22_tile_mult.w_base_idx * w_tile, output22_tile_mult.c_base_idx};
			output23_tile_base = {output23_tile_mult.h_base_idx * h_tile, output23_tile_mult.w_base_idx * w_tile, output23_tile_mult.c_base_idx};

			// std::cout<<"sdsds "<<input11_tile_base.w_base_idx<<std::endl;
			conv_relu_forward_tiled(input11, output11, conv1_filter, conv1_conf, input11_conf, 
										input11_tiled_conf, output11_tiled_conf, input11_tile_base, output11_tile_base);
			pool_forward_tiled(output11, output13, input12_tiled_conf, pool1_conf, input13_tile_base, output13_tile_base);
			
			conv_relu_forward_tiled(output13, output21, conv2_filter, conv2_conf, input21_conf,
										input21_tiled_conf, output21_tiled_conf, input21_tile_base, input21_tile_base);
			pool_forward_tiled(output21, output23, input22_tiled_conf, pool2_conf, input23_tile_base, output23_tile_base);
		}
	}
}