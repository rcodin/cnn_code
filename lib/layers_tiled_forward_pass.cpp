#include <layers.hpp>
#include <cmath>
#include <tiling.hpp>

void conv_forward_tiled(float ***in, float ***out, float ****filter, Conv_conf conv_conf, 
					Data_conf input_conf, Data_conf output_conf, tile_idx_conf input_tile_conf, tile_idx_conf ouptut_tile_conf) {	
	int in_h = input_conf.h;
	int in_w = input_conf.w;
	int in_c = input_conf.c;

	int out_h = output_conf.h;
	int out_w = output_conf.w;
	int out_c = output_conf.c;


	//whole convolution layer
	#pragma omp parallel
	for (int h_idx = 0; h_idx <  out_h; h_idx++) {
		for (int w_idx = 0; w_idx < out_w; w_idx++) {
			for (int c_idx = 0; c_idx < out_c; c_idx++) {
				//for each output point
				int h_in_idx = input_tile_conf.h_base_idx + h_idx;
				int w_in_idx = input_tile_conf.w_base_idx + w_idx; 

				int h_out_idx = ouptut_tile_conf.h_base_idx + h_idx;
				int w_out_idx = ouptut_tile_conf.w_base_idx + w_idx;
				
				
				// std::cout<<conv_conf.h<<conv_conf.w<<conv_conf.in_c<<std::endl;
				float elem = 0.0f;
				for (int i = 0; i < conv_conf.h; i++) {
					for (int j = 0; j < conv_conf.w; j++) {
						for (int k = 0; k < in_c; k++) {
							// if (input_tile_conf.w_base_idx > 0)
							// 	std::cout<<"dsddss : "<<w_in_idx<<" "<<w_out_idx<<std::endl;
							elem += in[h_in_idx + i][w_in_idx + j][k] * filter[c_idx][i][j][k];
							
						}
					}
				}
				out[h_out_idx][w_out_idx][c_idx] = elem;
			}
		}
	}
	// std::cout<<"conv_tiled"<<std::endl;
}

//orig_conf is the size of original input (before tiling)
void conv_relu_forward_tiled(float ***in, float ***out, float ****filter, Conv_conf conv_conf, Data_conf orig_conf,
					Data_conf input_conf, Data_conf output_conf, tile_idx_conf input_tile_conf, tile_idx_conf ouptut_tile_conf) {	
	int in_h = input_conf.h;
	int in_w = input_conf.w;
	int in_c = input_conf.c;

	int out_h = output_conf.h;
	int out_w = output_conf.w;
	int out_c = output_conf.c;

	#pragma omp parallel for
	for (int h_idx = 0; h_idx <  out_h; h_idx++) {
		for (int w_idx = 0; w_idx < out_w; w_idx++) {
			for (int c_idx = 0; c_idx < out_c; c_idx++) {
				//for each output point
				int h_in_idx = input_tile_conf.h_base_idx + h_idx;
				int w_in_idx = input_tile_conf.w_base_idx + w_idx; 

				int h_out_idx = ouptut_tile_conf.h_base_idx + h_idx;
				int w_out_idx = ouptut_tile_conf.w_base_idx + w_idx;
				
				
				// std::cout<<conv_conf.h<<conv_conf.w<<conv_conf.in_c<<std::endl;
				float elem = 0.0f;
				for (int i = 0; i < conv_conf.h; i++) {
					for (int j = 0; j < conv_conf.w; j++) {
						for (int k = 0; k < in_c; k++) {
							// std::cerr<<"w_in_idx : "<<(w_in_idx)<<std::endl;
							if ((h_in_idx + i) < orig_conf.h && (w_in_idx + j) < orig_conf.w)
								elem += in[h_in_idx + i][w_in_idx + j][k] * filter[c_idx][i][j][k];							
						}
					}
				}
				// std::cerr<<"w_in_idx : "<<(out_w)<<std::endl;
				if (elem > 0)
					out[h_out_idx][w_out_idx][c_idx] = elem;
				else
					out[h_out_idx][w_out_idx][c_idx] = 0;
			}
		}
	}	
}

void pool_forward_tiled(float ***in, float ***out, Data_conf input_conf, Pool_conf pool_conf,
					tile_idx_conf input_tile_conf, tile_idx_conf output_tile_conf) {
	//initialize out if not already initialized
	for (int h_idx = 0; h_idx < (input_conf.h); h_idx++) {
		for (int w_idx = 0; w_idx < (input_conf.w); w_idx++) {
			for (int c_idx = 0; c_idx < (input_conf.c); c_idx++) {
				// std::cout<<"pool"<<std::endl;
				int h_in_idx = input_tile_conf.h_base_idx + h_idx;
				int w_in_idx = input_tile_conf.w_base_idx + w_idx; 

				int h_out_idx = output_tile_conf.h_base_idx + h_idx/pool_conf.h;
				int w_out_idx = output_tile_conf.w_base_idx + w_idx/pool_conf.w;

				out[h_out_idx][w_out_idx][c_idx] = 
					std::fmax(out[h_out_idx][w_out_idx][c_idx], in[h_in_idx][w_in_idx][c_idx]);
			}
		}
	}
	// std::cout<<"pool_tiled"<<std::endl;
}

void relu_forward_tiled(float ***in, float ***out, Data_conf input_conf,
					tile_idx_conf input_tile_conf, tile_idx_conf output_tile_conf) {
	for (int h_idx = 0; h_idx < (input_conf.h); h_idx++) {
		for (int w_idx = 0; w_idx < (input_conf.w); w_idx++) {
			for (int c_idx = 0; c_idx < (input_conf.c); c_idx++) {

				int h_in_idx = input_tile_conf.h_base_idx + h_idx;
				int w_in_idx = input_tile_conf.w_base_idx + w_idx; 

				int h_out_idx = output_tile_conf.h_base_idx + h_idx;
				int w_out_idx = output_tile_conf.w_base_idx + w_idx;

				// std::cout<<"relu"<<std::endl;
				out[h_out_idx][w_out_idx][c_idx] = std::fmax(in[h_in_idx][w_in_idx][c_idx], 0);
			}
		}
	}
	// std::cout<<"relu_tiled"<<std::endl;
}