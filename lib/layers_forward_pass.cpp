#include <layers.hpp>
#include <cmath>
#include <utils.hpp>
#include <im2col.hpp>
#include <mkl.h>

void conv_forward(float ***in, float ***out, float ****filter, Conv_conf conv_conf, Data_conf input_conf, Data_conf output_conf) {	
	int in_h = input_conf.h;
	int in_w = input_conf.w;
	int in_c = input_conf.c;

	int out_h = output_conf.h;
	int out_w = output_conf.w;
	int out_c = output_conf.c;

	#pragma omp parallel
	for (int h_idx = 0; h_idx < out_h; h_idx++) {
		for (int w_idx = 0; w_idx < out_w; w_idx++) {
			for (int c_idx = 0; c_idx < out_c; c_idx++) {
				//for each output point
				
				// std::cout<<conv_conf.h<<conv_conf.w<<conv_conf.in_c<<std::endl;
				float elem = 0.0f;
				for (int i = 0; i < conv_conf.h; i++) {
					for (int j = 0; j < conv_conf.w; j++) {
						for (int k = 0; k < in_c; k++) {	
							elem += in[h_idx + i][w_idx + j][k] * filter[c_idx][i][j][k];
						}
					}
				}
				out[h_idx][w_idx][c_idx] = elem;
			}
		}
	}
}
void conv_im2col(float *in, float *out, float *filter, Conv_conf conv_conf,
					Data_conf input_conf, Data_conf output_conf) {
	int pad = conv_conf.h / 2;
	int channels = input_conf.c;
	int height = input_conf.h;
	int width = input_conf.w;
	int ksize = conv_conf.h;
	int stride = 1;

	float *patch_mat = (float *)mkl_malloc(input_conf.h * input_conf.w * input_conf.c * 
		conv_conf.h * conv_conf.w * sizeof(float), 32);
	
	im2col_cpu(in, channels, height, width, ksize, stride, pad, patch_mat);
	
	//gemmm
	//use intel cblas gemm to start with



}

void conv_relu_forward(float ***in, float ***out, float ****filter, Conv_conf conv_conf,
					Data_conf input_conf, Data_conf output_conf) {	
	int in_h = input_conf.h;
	int in_w = input_conf.w;
	int in_c = input_conf.c;

	int out_h = output_conf.h;
	int out_w = output_conf.w;
	int out_c = output_conf.c;

	print_conf_cfg(conv_conf, input_conf, output_conf);
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	// long long count = 0;
	// int count = 0;
	#if !TILE
		#pragma omp parallel
		for (int h_idx = 0; h_idx < out_h; h_idx++) {
			for (int w_idx = 0; w_idx < out_w; w_idx++) {
				for (int c_idx = 0; c_idx < out_c; c_idx++) {
					//for each output point
					
					// std::cout<<conv_conf.h<<conv_conf.w<<conv_conf.in_c<<std::endl;
					float elem = 0.0f;
					for (int i = 0; i < conv_conf.h; i++) {
						for (int j = 0; j < conv_conf.w; j++) {
							for (int k = 0; k < in_c; k++) {	
								elem += in[h_idx + i][w_idx + j][k] * filter[c_idx][i][j][k];
							}
						}
					}
					out[h_idx][w_idx][c_idx] = std::fmax(0, elem);
				}
			}
		}
	#else
    if (output_conf.c <= 64) {
	    #pragma omp parallel
		for (int h_idx_out = 0; h_idx_out < out_h; h_idx_out += 16) {
			for (int w_idx_out = 0; w_idx_out < out_w; w_idx_out += 16) {
				for (int c_idx = 0; c_idx < out_c; c_idx++) {
					//for each output point
					for (int h_idx = h_idx_out; h_idx < (h_idx_out + 16); h_idx++) {
						for (int w_idx = w_idx_out; w_idx < (w_idx_out + 16); w_idx++) {
					// std::cout<<conv_conf.h<<conv_conf.w<<conv_conf.in_c<<std::endl;
							float elem = 0.0f;
							for (int i = 0; i < conv_conf.h; i++) {
								for (int j = 0; j < conv_conf.w; j++) {
									for (int k = 0; k < in_c; k++) {
										// count++;
										elem += in[h_idx + i][w_idx + j][k] * filter[c_idx][i][j][k];
									}
								}
							}
							out[h_idx][w_idx][c_idx] = std::fmax(0, elem);
						}
					}
				}
			}
		}
	}
	// high_resolution_clock::time_point t2 = high_resolution_clock::now();
	// auto duration = duration_cast<microseconds>( t2 - t1 ).count();


	// std::cout<<"Time taken conv type 1 :"<< duration<<std::endl;

	// t1 = high_resolution_clock::now();
	else {
		#pragma omp parallel
		for (int h_idx_out = 0; h_idx_out < out_h; h_idx_out += 7) {
			for (int w_idx_out = 0; w_idx_out < out_w; w_idx_out += 7) {
				for (int c_idx = 0; c_idx < out_c; c_idx++) {
					//for each output point
					for (int h_idx = h_idx_out; h_idx < (h_idx_out + 7); h_idx++) {
						for (int w_idx = w_idx_out; w_idx < (w_idx_out + 7); w_idx++) {
					// std::cout<<conv_conf.h<<conv_conf.w<<conv_conf.in_c<<std::endl;
							float elem = 0.0f;
							for (int i = 0; i < conv_conf.h; i++) {
								for (int j = 0; j < conv_conf.w; j++) {
									for (int k = 0; k < in_c; k++) {
										// count++;
										elem += in[h_idx + i][w_idx + j][k] * filter[c_idx][i][j][k];
									}
								}
							}
							out[h_idx][w_idx][c_idx] = std::fmax(0, elem);
						}
					}
				}
			}
		}
	}
	// std::cout<<"conv : "<<count<<std::endl;
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<milliseconds>( t2 - t1 ).count();

	std::cout<<"Time taken in milliseconds : "<<duration<<std::endl<<std::endl;
	#endif
}

void pool_forward(float ***in, float ***out, Data_conf input_conf, Pool_conf pool_conf) {
	high_resolution_clock::time_point t1 = high_resolution_clock::now();
	//initialize out if not already initialized
	for (int h_idx = 0; h_idx < input_conf.h; h_idx++) {
		for (int w_idx = 0; w_idx < input_conf.w; w_idx++) {
			for (int c_idx = 0; c_idx < input_conf.c; c_idx++) {
				out[h_idx/pool_conf.h][w_idx/pool_conf.w][c_idx] = 
					std::fmax(out[h_idx/pool_conf.h][w_idx/pool_conf.w][c_idx], in[h_idx][w_idx][c_idx]);
			}
		}
	}
	high_resolution_clock::time_point t2 = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>( t2 - t1 ).count();

	std::cout<<"Time taken in microseconds : "<<duration<<std::endl<<std::endl;
	// std::cout<<"pool"<<std::endl;
}


void relu_forward(float ***in, float ***out, Data_conf input_conf) {
	for (int i = 0; i < input_conf.h; i++) {
		for (int j = 0; j < input_conf.w; j++) {
			for (int k = 0; k < input_conf.c; k++) {
				out[i][j][k] = std::fmax(in[i][j][k], 0);
			}
		}
	}
}

void linearize_conv(float ***in, float *out, float **filter, Data_conf input_conf, int out_size) {
	// int out_size = output_conf.h;
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
			out[j] += in[i] * filter[i][j];
		}
	}
	std::cout<<"fc2"<<std::endl;
}

void fc_softmax_forward(float *in, float *out, float **filter,int input_size, int output_size) {
	for (int i = 0; i < input_size; i++) {
		for (int j = 0; j < output_size; j++) {
			// std::cout<<"fc2"<<std::endl;
			out[j] += in[i] * filter[i][j];
		}
	}
	std::cout<<"fc2"<<std::endl;
}