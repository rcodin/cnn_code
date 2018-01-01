#include "layers.hpp"

void conv_forward(float ***in, float ***out, void *filter, Conv_conf conv_conf) {
	
	int in_h = conv_conf.h;
	int in_w = conv_conf.w;
	int in_c = conv_conf.in_c;

	int out_h = conv_conf.h - conv_conf.f_h + 1;
	int out_w = conv_conf.w - conv_conf.f_w + 1;
	int out_c = conv_conf.out_c;

	//whole convolution layer
	for (int h_idx = 0; h_idx < out_h; h_idx++) {
		for (int w_idx = 0; w_idx < w_idx; w_idx++) {
			for (int c_idx = 0; c_idx < out_c; c_idx++) {
				//for each output point

				float elem = 0.0f;
				for (int i = 0; i < conv_conf.f_h; i++) {
					for (int j = 0; j < conv_conf.f_w; j++) {
						for (int k = 0; k < conv_conf.in_c; k++) {
							elem += in[h_idx + i][w_idx + j][k];
						}
					}
				}
				out[out_h][out_w][out_c] = elem;
			}
		}
	}
	// return 0;

}

void pool_forward(void *in, void *out, Input_conf input_conf,Pool_conf pool_conf) {

}

void relu_forward(void *in, void *out, Input_conf input_conf) {

}