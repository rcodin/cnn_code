/*
Neural Network with 2 tiled and fused layers
*/

#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <layers.hpp>
#include <tiling.hpp>
#include <utils.hpp>

using namespace std;

int main()
{
	//conv->relu->pool conv->relu->pool conv->relu->pool fc->fc
	
	//load network weights
	size_t bytes = sizeof(float);
	float ****conv11_filter = (float ****)alloc_4D(64, 3, 3, 3, bytes);
	float ****conv12_filter = (float ****)alloc_4D(128, 3, 3, 64, bytes);
	
	//create input 
	//conv1->relu->pool

	//Conv1
	Conv_conf conv11_conf = {3, 3};
	
	Data_conf input11_conf = {224, 224, 3};
	Data_conf output11_conf = {224, 224, 64};

	
	//conv2
	Conv_conf conv12_conf = {3, 3};

	Data_conf input12_conf = {224, 224, 64};
	Data_conf output12_conf = {224, 224, 64};


	Data_conf input11_tiled_conf = {228, 228, 3};
	Data_conf output11_tiled_conf = {226, 226, 64};

	Data_conf input12_tiled_conf = {226, 226, 16};
	Data_conf output12_tiled_conf = {224, 224, 64};

	//pool1
	
	tile_idx_conf input11_tile_mult, input12_tile_mult;
	tile_idx_conf output11_tile_mult, output12_tile_mult;
	


	output12_tile_mult = {output12_tiled_conf.h, output12_tiled_conf.w, output12_tiled_conf.c};
	input12_tile_mult = {output12_tile_mult.h_base_idx, output12_tile_mult.w_base_idx, output12_tile_mult.c_base_idx};

	output11_tile_mult = {input12_tile_mult.h_base_idx, input12_tile_mult.w_base_idx, input12_tile_mult.c_base_idx};
	input11_tile_mult = {output11_tile_mult.h_base_idx, output11_tile_mult.w_base_idx, input11_tiled_conf.c};

	int h_num_tiles = 1;
	int w_num_tiles = 1;

	tile_idx_conf input11_tile_base, input12_tile_base;
	tile_idx_conf output11_tile_base, output12_tile_base;

	float ***input11 = (float ***)alloc_3D(input11_tiled_conf.h, input11_tiled_conf.w, input11_conf.c, bytes);
	float ***output11 = (float ***)alloc_3D(output11_tiled_conf.h, output11_tiled_conf.w, output11_conf.c, bytes);
	float ***output12 = (float ***)alloc_3D(output12_tiled_conf.h, output12_tiled_conf.w, output12_tiled_conf.c, bytes);


	// #pragma omp parallel for
	for (int h_tile = 0; h_tile < h_num_tiles; h_tile++) {
		for (int w_tile = 0; w_tile < h_num_tiles; w_tile++) {
			//conv1->relu->pool
			input11_tile_base = {0, 0, input11_tile_mult.c_base_idx};
			input12_tile_base = {0, 0, input12_tile_mult.c_base_idx};
			
			output11_tile_base = {0, 0, output11_tile_mult.c_base_idx};
			output12_tile_base = {0, 0, output12_tile_mult.c_base_idx};
			
			conv_relu_forward_tiled_parallel(input11, output11, conv11_filter, conv11_conf, input11_conf, input11_tiled_conf, output11_tiled_conf, input11_tile_base, output11_tile_base);
			conv_relu_forward_tiled_parallel(output11, output12, conv12_filter, conv12_conf, input12_conf, input12_tiled_conf, output12_tiled_conf, input12_tile_base, input12_tile_base);
		}
	}
}