#include <layers.hpp>
#include <cmath>


void conv_forward(float ***in, float ***out, float ****filter, Conv_conf conv_conf, Data_conf input_conf, Data_conf output_conf) {	
	int in_h = input_conf.h;
	int in_w = input_conf.w;
	int in_c = input_conf.c;

	int out_h = conv_conf.h - conv_conf.f_h + 1;
	int out_w = conv_conf.w - conv_conf.f_w + 1;
	int out_c = conv_conf.out_c;

	//whole convolution layer
	for (int h_idx = 0; h_idx < out_h; h_idx++) {
		for (int w_idx = 0; w_idx < out_w; w_idx++) {
			for (int c_idx = 0; c_idx < out_c; c_idx++) {
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

void pool_forward(float ***in, float ***out, Input_conf input_conf, Pool_conf pool_conf) {
	//initialize out if not already initialized
	for (int h_idx = 0; h_idx < input_conf.h; h_idx++) {
		for (int w_idx = 0; w_idx < input_conf.w; w_idx++) {
			for (int c_idx = 0; c_idx < input_conf.c; c_idx++) {
				// std::cout<<"pool"<<std::endl;
				out[h_idx/pool_conf.h][w_idx/pool_conf.w][c_idx] = 
					std::fmax(out[h_idx/pool_conf.h][w_idx/pool_conf.w][c_idx], in[h_idx][w_idx][c_idx]);
			}
		}
	}

}

void relu_forward(float ***in, float ***out, Input_conf input_conf) {
	for (int i = 0; i < input_conf.h; i++) {
		for (int j = 0; j < input_conf.w; j++) {
			for (int k = 0; k < input_conf.c; k++) {
				// std::cout<<"relu"<<std::endl;
				out[i][j][k] = std::fmax(in[i][j][k], 0);
			}
		}
	}	
}

void linearize_conv(float ***in, float *out, float **filter, Input_conf input_conf, Input_conf output_conf) {
	int out_size = output_conf.h;
	int i_mult = input_conf.w * input_conf.c;
	int j_mult = input_conf.c;

	for (int out_idx = 0; out_idx < out_size; out_idx++) {
		float out_elem = 0.0f;
		for (int i = 0; i < input_conf.h; i++) {
			for (int j = 0; j < input_conf.w; j++) {
				for (int k = 0; k < input_conf.c; k++) {
					out_elem += in[i][j][k] * filter[out_idx][i * i_mult + j * j_mult + k];	
				}
			}
		}
		out[out_idx] = out_elem;
	}
}

void fc_forward(float *in, float *out, float **filter,int input_size, int output_size) {
	for (int i = 0; i < input_size; i++) {
		for (int j = 0; j < output_size; j++) {
			std::cout<<"fc2"<<std::endl;
			out[j] = in[i] * filter[i][j];
		}
	}
}