#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <layers.hpp>
#include <tiling.hpp>
#include <utils.hpp>
#include <mkl.h>

int main() {
	//224x224x3 Conv
		//load network weights
	size_t bytes = sizeof(float);
	int alignment = bytes * 8;
	//create input 
	//conv1->relu->pool

		//Conv11
		Conv_conf conv11_conf = {3, 3};
		Data_conf input11_conf = {224, 224, 3};
		Data_conf output11_conf = {224, 224, 64};

		
		//conv12
		Conv_conf conv12_conf = {3, 3};
		Data_conf input12_conf = {224, 224, 64};
		Data_conf output12_conf = {224, 224, 64};

		//Pool1
		Pool_conf pool1_conf = {2, 2};
		Data_conf input13_conf = {224, 224, 64};
		Data_conf output13_conf = {112, 112, 64};

		//Conv21
		Conv_conf conv21_conf = {3, 3};	
		Data_conf input21_conf = {112, 112, 64};
		Data_conf output21_conf = {112, 112, 128};

		//Conv22
		Conv_conf conv22_conf = {3, 3};
		Data_conf input22_conf = {112, 112, 128};
		Data_conf output22_conf = {112, 112, 128};

		//Pool2	
		Pool_conf pool2_conf = {2, 2};
		Data_conf input23_conf = {112, 112, 128};
		Data_conf output23_conf = {56, 56, 128};


		//Conv31
		Conv_conf conv31_conf = {3, 3};	
		Data_conf input31_conf = {56, 56, 128};
		Data_conf output31_conf = {56, 56, 256};

		//Conv32
		Conv_conf conv32_conf = {3, 3};	
		Data_conf input32_conf = {56, 56, 256};
		Data_conf output32_conf = {56, 56, 256};

		//Conv33
		Conv_conf conv33_conf = {3, 3};	
		Data_conf input33_conf = {56, 56, 256};
		Data_conf output33_conf = {56, 56, 256};

		//Pool3
		Pool_conf pool3_conf = {2, 2};
		Data_conf input34_conf = {56, 56, 256};
		Data_conf output34_conf = {28, 28, 256};

		//Conv41
		Conv_conf conv41_conf = {3, 3};	
		Data_conf input41_conf = {28, 28, 256};
		Data_conf output41_conf = {28, 28, 512};

		//Conv42
		Conv_conf conv42_conf = {3, 3};	
		Data_conf input42_conf = {28, 28, 512};
		Data_conf output42_conf = {28, 28, 512};

		//Conv43
		Conv_conf conv43_conf = {3, 3};	
		Data_conf input43_conf = {28, 28, 512};
		Data_conf output43_conf = {28, 28, 512};

		//Pool4
		Pool_conf pool4_conf = {2, 2};
		Data_conf input44_conf = {28, 28, 512};
		Data_conf output44_conf = {14, 14, 512};

		//Conv51
		Conv_conf conv51_conf = {3, 3};	
		Data_conf input51_conf = {14, 14, 512};
		Data_conf output51_conf = {14, 14, 512};

		//Conv52
		Conv_conf conv52_conf = {3, 3};	
		Data_conf input52_conf = {14, 14, 512};
		Data_conf output52_conf = {14, 14, 512};

		//Conv53
		Conv_conf conv53_conf = {3, 3};	
		Data_conf input53_conf = {14, 14, 512};
		Data_conf output53_conf = {14, 14, 512};

		//Pool5
		Pool_conf pool5_conf = {3, 3};
		Data_conf input54_conf = {14, 14, 512};
		Data_conf output54_conf = {7, 7, 512};	

		//fc1 flattening
		Data_conf input6_conf = {7, 7, 512};
		int output6_conf = 4096;
		
		//fc2
		int input7_conf = 4096;
		int output7_conf = 4096;
		
		//fc3_softmax
		int input8_conf = 4096;
		int output8_conf = 1000;

	float *input11 = (float *)mkl_malloc(input11_conf.h * input11_conf.w * input11_conf.c * bytes, alignment);
	float *output11 = (float *)mkl_malloc(output11_conf.h * output11_conf.w * output11_conf.c*bytes, alignment);
	float *output12 = (float *)mkl_malloc(output12_conf.h * output12_conf.w * output12_conf.c*bytes, alignment);
	float *output13 = (float *)mkl_malloc(output13_conf.h * output13_conf.w * output13_conf.c*bytes, alignment);

	float *output21 = (float *)mkl_malloc(output21_conf.h * output21_conf.w * output21_conf.c*bytes, alignment);
	float *output22 = (float *)mkl_malloc(output22_conf.h * output22_conf.w * output22_conf.c*bytes, alignment);
	float *output23 = (float *)mkl_malloc(output23_conf.h * output23_conf.w * output23_conf.c*bytes, alignment);

	float *output31 = (float *)mkl_malloc(output31_conf.h * output31_conf.w * output31_conf.c*bytes, alignment);
	float *output32 = (float *)mkl_malloc(output32_conf.h * output32_conf.w * output32_conf.c*bytes, alignment);
	float *output33 = (float *)mkl_malloc(output33_conf.h * output33_conf.w * output33_conf.c*bytes, alignment);
	float *output34 = (float *)mkl_malloc(output34_conf.h * output34_conf.w * output34_conf.c*bytes, alignment);

	float *output41 = (float *)mkl_malloc(output41_conf.h * output41_conf.w * output41_conf.c*bytes, alignment);
	float *output42 = (float *)mkl_malloc(output42_conf.h * output42_conf.w * output42_conf.c*bytes, alignment);
	float *output43 = (float *)mkl_malloc(output43_conf.h * output43_conf.w * output43_conf.c*bytes, alignment);
	float *output44 = (float *)mkl_malloc(output44_conf.h * output44_conf.w * output44_conf.c*bytes, alignment);

	float *output51 = (float *)mkl_malloc(output51_conf.h * output51_conf.w * output51_conf.c*bytes, alignment);
	float *output52 = (float *)mkl_malloc(output52_conf.h * output52_conf.w * output52_conf.c*bytes, alignment);
	float *output53 = (float *)mkl_malloc(output53_conf.h * output53_conf.w * output53_conf.c*bytes, alignment);
	float *output54 = (float *)mkl_malloc(output54_conf.h * output54_conf.w * output54_conf.c*bytes, alignment);

	float *output6 = (float *)alloc_1D(output6_conf, bytes);
	float *output7 = (float *)alloc_1D(output7_conf, bytes);
	float *output8 = (float *)alloc_1D(output8_conf, bytes);


	//allocating filers
	float *conv11_weights = (float *)mkl_malloc(output11_conf.c * conv11_conf.h * conv11_conf.w * input11_conf.c * bytes,  alignment);
	float *conv12_weights = (float *)mkl_malloc(output12_conf.c * conv12_conf.h * conv12_conf.w * input12_conf.c * bytes,  alignment);

	float *conv21_weights = (float *)mkl_malloc(output21_conf.c * conv21_conf.h * conv21_conf.w * input21_conf.c * bytes,  alignment);
	float *conv22_weights = (float *)mkl_malloc(output22_conf.c * conv22_conf.h * conv22_conf.w * input22_conf.c * bytes,  alignment);

	float *conv31_weights = (float *)mkl_malloc(output31_conf.c * conv31_conf.h * conv31_conf.w * input31_conf.c * bytes,  alignment);
	float *conv32_weights = (float *)mkl_malloc(output32_conf.c * conv32_conf.h * conv32_conf.w * input32_conf.c * bytes,  alignment);
	float *conv33_weights = (float *)mkl_malloc(output33_conf.c * conv32_conf.h * conv32_conf.w * input33_conf.c * bytes,  alignment);

	float *conv41_weights = (float *)mkl_malloc(output41_conf.c * conv41_conf.h * conv41_conf.w * input41_conf.c * bytes,  alignment);
	float *conv42_weights = (float *)mkl_malloc(output42_conf.c * conv42_conf.h * conv42_conf.w * input42_conf.c * bytes,  alignment);
	float *conv43_weights = (float *)mkl_malloc(output43_conf.c * conv43_conf.h * conv43_conf.w * input43_conf.c * bytes,  alignment);

	float *conv51_weights = (float *)mkl_malloc(output51_conf.c * conv51_conf.h * conv51_conf.w * input51_conf.c * bytes,  alignment);
	float *conv52_weights = (float *)mkl_malloc(output52_conf.c * conv52_conf.h * conv52_conf.w * input52_conf.c * bytes,  alignment);
	float *conv53_weights = (float *)mkl_malloc(output53_conf.c * conv53_conf.h * conv53_conf.w * input53_conf.c * bytes,  alignment);

	//allocating biases

	float *conv11_biases = (float *)mkl_malloc(output11_conf.c  * bytes,  alignment);
	float *conv12_biases = (float *)mkl_malloc(output12_conf.c  * bytes,  alignment);

	float *conv21_biases = (float *)mkl_malloc(output21_conf.c  * bytes,  alignment);
	float *conv22_biases = (float *)mkl_malloc(output22_conf.c  * bytes,  alignment);

	float *conv31_biases = (float *)mkl_malloc(output31_conf.c  * bytes,  alignment);
	float *conv32_biases = (float *)mkl_malloc(output32_conf.c  * bytes,  alignment);
	float *conv33_biases = (float *)mkl_malloc(output33_conf.c  * bytes,  alignment);

	float *conv41_biases = (float *)mkl_malloc(output41_conf.c  * bytes,  alignment);
	float *conv42_biases = (float *)mkl_malloc(output42_conf.c  * bytes,  alignment);
	float *conv43_biases = (float *)mkl_malloc(output43_conf.c  * bytes,  alignment);

	float *conv51_biases = (float *)mkl_malloc(output51_conf.c  * bytes,  alignment);
	float *conv52_biases = (float *)mkl_malloc(output52_conf.c  * bytes,  alignment);
	float *conv53_biases = (float *)mkl_malloc(output53_conf.c  * bytes,  alignment);




	float **fc1_filter = (float **)alloc_2D(input6_conf.h * input6_conf.w * input6_conf.c, output6_conf , bytes);
	float **fc2_filter = (float **)alloc_2D(input7_conf, output7_conf, bytes);
	float **fc3_filter = (float **)alloc_2D(input8_conf, output8_conf, bytes);
	

	//Group 1
	conv_im2col(input11, output11, conv11_weights,conv11_biases, conv11_conf, input11_conf, output11_conf);
	conv_im2col(output11, output12, conv12_weights,conv12_biases, conv12_conf, input12_conf, output12_conf);
	// pool_forward(output12, output13, input13_conf, pool1_conf);

	// //Group 2
	conv_im2col(output13, output21, conv21_weights,conv21_biases, conv21_conf, input21_conf, output21_conf);
	conv_im2col(output21, output22, conv22_weights,conv22_biases, conv22_conf, input22_conf, output22_conf);
	// pool_forward(output22, output23, input23_conf, pool2_conf);

	// //Group 3
	conv_im2col(output23, output31, conv31_weights,conv31_biases, conv31_conf, input31_conf, output31_conf);
	conv_im2col(output31, output32, conv32_weights,conv32_biases, conv32_conf, input32_conf, output32_conf);
	conv_im2col(output32, output33, conv33_weights,conv33_biases, conv33_conf, input33_conf, output33_conf);
	// pool_forward(output33, output34, input34_conf, pool3_conf);
	
	// //Group 4
	conv_im2col(output34, output41, conv41_weights,conv41_biases, conv41_conf, input41_conf, output41_conf);
	conv_im2col(output41, output42, conv42_weights,conv42_biases, conv42_conf, input42_conf, output42_conf);
	conv_im2col(output42, output43, conv43_weights,conv43_biases, conv43_conf, input43_conf, output43_conf);
	// pool_forward(output43, output44, input44_conf, pool4_conf);

	// //Group 5
	conv_im2col(output44, output51, conv51_weights,conv51_biases, conv51_conf, input51_conf, output51_conf);
	conv_im2col(output51, output52, conv52_weights,conv52_biases, conv52_conf, input52_conf, output52_conf);
	conv_im2col(output52, output53, conv53_weights,conv53_biases, conv53_conf, input53_conf, output53_conf);
	// pool_forward(output53, output54, input54_conf, pool5_conf);

	//fc1
	// linearize_conv(output54, output6, fc1_filter, input6_conf, output6_conf);
	
	//fc2
	fc_forward(output6, output7, fc2_filter, input7_conf, output7_conf);
	
	//fc3
	fc_forward(output7, output8, fc3_filter, input8_conf, output8_conf);

	// Conv_conf conv11_conf = {3, 3};
	// Data_conf input11_conf = {224, 224, 3};
	// Data_conf output11_conf = {224, 224, 64};

	// float *in = (float *)mkl_malloc(input11_conf.h * output11_conf.w * input11_conf.c * sizeof(float), 32);
	// float *out = (float *)mkl_malloc(output11_conf.h * output11_conf.w * output11_conf.c * sizeof(float), 32);
	// float *weights = (float *)mkl_malloc(output11_conf.c * conv11_conf.h * conv11_conf.w * input11_conf.c * sizeof(float), 32);
	// float *biases = (float *)mkl_malloc(output11_conf.c * sizeof(float), 32);
	
	// conv_im2col(in, out, weights, biases,conv11_conf, input11_conf, output11_conf);


	return 0;
}