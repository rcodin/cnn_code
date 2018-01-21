#include <layers.hpp>
#include <cmath>
#include <tiling.hpp>

void conv_forward_tiled(float ***in, float ***out, float ****filter, Conv_conf conv_conf, 
					Data_conf input_conf, Data_conf output_conf, tile_idx_conf tile_conf) {	
	int in_h = input_conf.h;
	int in_w = input_conf.w;
	int in_c = input_conf.c;

	int out_h = output_conf.h;
	int out_w = output_conf.w;
	int out_c = output_conf.c;

	//whole convolution layer
	for (int h_idx = tile_conf.h_base_idx; h_idx < (tile_conf.h_base_idx + out_h); h_idx++) {
		for (int w_idx = tile_conf.w_base_idx; w_idx < (tile_conf.w_base_idx + out_w); w_idx++) {
			for (int c_idx = tile_conf.c_base_idx; c_idx < (tile_conf.c_base_idx + out_c); c_idx++) {
				//for each output point
				
				// std::cout<<conv_conf.h<<conv_conf.w<<conv_conf.in_c<<std::endl;
				float elem = 0.0f;
				for (int i = 0; i < conv_conf.h; i++) {
					for (int j = 0; j < conv_conf.w; j++) {
						for (int k = 0; k < in_c; k++) {	
							elem += in[h_idx + i][w_idx + j][k] * filter[i][j][k][c_idx];
						}
					}
				}
				out[h_idx][w_idx][c_idx] = elem;
			}
		}
	}
	std::cout<<"mult"<<std::endl;
}

void pool_forward_tiled(float ***in, float ***out, Data_conf input_conf, Pool_conf pool_conf,
					tile_idx_conf tile_conf) {
	//initialize out if not already initialized
	for (int h_idx = tile_conf.h_base_idx; h_idx < (tile_conf.h_base_idx + input_conf.h); h_idx++) {
		for (int w_idx = tile_conf.w_base_idx; w_idx < (tile_conf.w_base_idx + input_conf.w); w_idx++) {
			for (int c_idx = tile_conf.c_base_idx; c_idx < (tile_conf.c_base_idx + input_conf.c); c_idx++) {
				// std::cout<<"pool"<<std::endl;
				out[h_idx/pool_conf.h][w_idx/pool_conf.w][c_idx] = 
					std::fmax(out[h_idx/pool_conf.h][w_idx/pool_conf.w][c_idx], in[h_idx][w_idx][c_idx]);
			}
		}
	}

}

void relu_forward_tiled(float ***in, float ***out, Data_conf input_conf,
					tile_idx_conf tile_conf) {
	for (int i = tile_conf.h_base_idx; i < (tile_conf.h_base_idx + input_conf.h); i++) {
		for (int j = tile_conf.w_base_idx; j < (tile_conf.w_base_idx + input_conf.w); j++) {
			for (int k = 0; k < (tile_conf.c_base_idx + input_conf.c); k++) {
				// std::cout<<"relu"<<std::endl;
				out[i][j][k] = std::fmax(in[i][j][k], 0);
			}
		}
	}	
}