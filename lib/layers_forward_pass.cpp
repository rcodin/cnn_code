#include <layers.hpp>
#include <cmath>
#include <utils.hpp>
#include <im2col.hpp>
#include <mkl.h>

void conv_forward(float *in, float *out, float *filter, Conv_conf conv_conf, Data_conf input_conf, Data_conf output_conf) {	
	int in_h = input_conf.h;
	int in_w = input_conf.w;
	int in_c = input_conf.c;

	int out_h = output_conf.h;
	int out_w = output_conf.w;
	int out_c = output_conf.c;

	for (int h_idx = 0; h_idx < out_h; h_idx++) {
		for (int w_idx = 0; w_idx < out_w; w_idx++) {
			for (int c_idx = 0; c_idx < out_c; c_idx++) {

				float elem = 0.0f;
				for (int i = 0; i < conv_conf.h; i++) {
					for (int j = 0; j < conv_conf.w; j++) {
						for (int k = 0; k < in_c; k++) {
							elem += in[((h_idx + i) * in_w + (w_idx + j)) * in_c + k] * 
									filter[((c_idx * conv_conf.h + i) * conv_conf.w + j) * in_c + k];
						}
					}
				}
				out[(h_idx * out_w + w_idx) * out_c + c_idx] = elem;
			}
		}
	}
}

void conv_im2col(float *in, float *out, float *weights, float *biases, Conv_conf conv_conf,
					Data_conf input_conf, Data_conf output_conf) {
	int pad = conv_conf.pad;
	int channels = input_conf.c;
	int height = input_conf.h;
	int width = input_conf.w;
	int ksize = conv_conf.h;
	int stride = conv_conf.stride;

	float *patch_mat = (float *)mkl_malloc(input_conf.h * input_conf.w * input_conf.c * 
		conv_conf.h * conv_conf.w * sizeof(float), 32);
	
	im2col_cpu(in, channels, height, width, ksize, stride, pad, patch_mat);
	

	//initialize output matrix 
	replicate_across_cols(biases, out, output_conf.c, output_conf.h * output_conf.w);
	//gemmm
	//use intel cblas gemm to start with
	// cblas_sgemm();
	CBLAS_LAYOUT layout = CblasRowMajor;
	CBLAS_TRANSPOSE transa = CblasNoTrans;
	CBLAS_TRANSPOSE transb = CblasNoTrans;
	MKL_INT m = output_conf.c;
	MKL_INT n = input_conf.h * input_conf.w;
	MKL_INT k = input_conf.c * conv_conf.h * conv_conf.w;
	float alpha = 1;
	const float *a = weights;
	MKL_INT lda = k;
	float *b = patch_mat;
	MKL_INT ldb = n;
	float beta = 1;
	float *c = out;
	MKL_INT ldc = n;
	cblas_sgemm(layout, transa, transb, m, n, k, alpha, a, lda, b, ldb, beta, c, ldc);
}

void pool_forward(float *in, float *out, Data_conf input_conf, Data_conf output_conf, Pool_conf pool_conf) {
	for (int h_idx = 0; h_idx < output_conf.h; h_idx++) {
		for (int w_idx = 0; w_idx < output_conf.w; w_idx++) {
			for (int c_idx = 0; c_idx < output_conf.c; c_idx++) {
				int input_hidx = h_idx * pool_conf.h; 
				int input_widx = w_idx * pool_conf.w;
				int input_cidx = c_idx;

				for (int i = 0; i < pool_conf.h; i++) {
					for (int j = 0; j < pool_conf.w; j++) {
						int in_idx = ((input_hidx + i) * input_conf.w + (input_widx + i)) * input_conf.c + input_cidx;
						int out_idx = (h_idx * output_conf.w + w_idx) * output_conf.c + c_idx;

						out[out_idx] =
							std::fmax(out[out_idx], in[in_idx]);
					}
				}
			}
		}
	}
}


void relu_forward(float *in, float *out, Data_conf input_conf) {
	for (int i = 0; i < input_conf.h; i++) {
		for (int j = 0; j < input_conf.w; j++) {
			for (int k = 0; k < input_conf.c; k++) {
				int idx = (i * input_conf.w + j) * input_conf.c + k;
				
				out[idx] = std::fmax(in[idx], 0);
			}
		}
	}
}

void fc_forward(float *in, float *out, float *filter, int input_size, int output_size) {
	for (int i = 0; i < input_size; i++) {
		for (int j = 0; j < output_size; j++) {
			out[j] += in[i] * filter[i * input_size +  j];
		}
	}
}


void softmax_forward(float *in, float *out, float *filter, int input_size, int output_size) {
	float tot = 0;

	for (int i = 0; i < input_size; i++) {
		out[i] = exp(in[i]);
		tot += out[i];
	}

	for (int j = 0; j < output_size; j++) {
		out[j] = out[j]/tot;
	}
}