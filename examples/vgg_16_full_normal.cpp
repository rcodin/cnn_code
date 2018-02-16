#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <layers.hpp>
#include <tiling.hpp>
#include <utils.hpp>
#include <chrono>

using namespace std;
using namespace std::chrono;

int main() {
	//conv->relu->pool conv->relu->pool conv->relu->pool fc->fc

	//load network weights
	size_t bytes = sizeof(float);
	
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

	float ***input11 = (float ***)alloc_3D(input11_conf.h + 2, input11_conf.w + 2, input11_conf.c, bytes);
	float ***output11 = (float ***)alloc_3D(output11_conf.h + 2, output11_conf.w + 2, output11_conf.c, bytes);
	float ***output12 = (float ***)alloc_3D(output12_conf.h + 2, output12_conf.w + 2, output12_conf.c, bytes);
	float ***output13 = (float ***)alloc_3D(output13_conf.h + 2, output13_conf.w + 2, output13_conf.c, bytes);

	float ***output21 = (float ***)alloc_3D(output21_conf.h + 2, output21_conf.w + 2, output21_conf.c, bytes);
	float ***output22 = (float ***)alloc_3D(output22_conf.h + 2, output22_conf.w + 2, output22_conf.c, bytes);
	float ***output23 = (float ***)alloc_3D(output23_conf.h + 2, output23_conf.w + 2, output23_conf.c, bytes);

	float ***output31 = (float ***)alloc_3D(output31_conf.h + 2, output31_conf.w + 2, output31_conf.c, bytes);
	float ***output32 = (float ***)alloc_3D(output32_conf.h + 2, output32_conf.w + 2, output32_conf.c, bytes);
	float ***output33 = (float ***)alloc_3D(output33_conf.h + 2, output33_conf.w + 2, output33_conf.c, bytes);
	float ***output34 = (float ***)alloc_3D(output34_conf.h + 2, output34_conf.w + 2, output34_conf.c, bytes);

	float ***output41 = (float ***)alloc_3D(output41_conf.h + 2, output41_conf.w + 2, output41_conf.c, bytes);
	float ***output42 = (float ***)alloc_3D(output42_conf.h + 2, output42_conf.w + 2, output42_conf.c, bytes);
	float ***output43 = (float ***)alloc_3D(output43_conf.h + 2, output43_conf.w + 2, output43_conf.c, bytes);
	float ***output44 = (float ***)alloc_3D(output44_conf.h + 2, output44_conf.w + 2, output44_conf.c, bytes);

	float ***output51 = (float ***)alloc_3D(output51_conf.h + 2, output51_conf.w + 2, output51_conf.c, bytes);
	float ***output52 = (float ***)alloc_3D(output52_conf.h + 2, output52_conf.w + 2, output52_conf.c, bytes);
	float ***output53 = (float ***)alloc_3D(output53_conf.h + 2, output53_conf.w + 2, output53_conf.c, bytes);
	float ***output54 = (float ***)alloc_3D(output54_conf.h + 2, output54_conf.w + 2, output54_conf.c, bytes);

	float *output6 = (float *)alloc_1D(output6_conf, bytes);
	float *output7 = (float *)alloc_1D(output7_conf, bytes);
	float *output8 = (float *)alloc_1D(output8_conf, bytes);


	//allocating filers
	float ****conv11_filter = (float ****)alloc_4D(output11_conf.c, conv11_conf.h, conv11_conf.w, input11_conf.c, bytes);
	float ****conv12_filter = (float ****)alloc_4D(output12_conf.c, conv12_conf.h, conv12_conf.w, input12_conf.c, bytes);

	float ****conv21_filter = (float ****)alloc_4D(output21_conf.c, conv21_conf.h, conv21_conf.w, input21_conf.c, bytes);
	float ****conv22_filter = (float ****)alloc_4D(output22_conf.c, conv22_conf.h, conv22_conf.w, input22_conf.c, bytes);

	float ****conv31_filter = (float ****)alloc_4D(output31_conf.c, conv31_conf.h, conv31_conf.w, input31_conf.c, bytes);
	float ****conv32_filter = (float ****)alloc_4D(output32_conf.c, conv32_conf.h, conv32_conf.w, input32_conf.c, bytes);
	float ****conv33_filter = (float ****)alloc_4D(output33_conf.c, conv32_conf.h, conv32_conf.w, input33_conf.c, bytes);

	float ****conv41_filter = (float ****)alloc_4D(output41_conf.c, conv41_conf.h, conv41_conf.w, input41_conf.c, bytes);
	float ****conv42_filter = (float ****)alloc_4D(output42_conf.c, conv42_conf.h, conv42_conf.w, input42_conf.c, bytes);
	float ****conv43_filter = (float ****)alloc_4D(output43_conf.c, conv43_conf.h, conv43_conf.w, input43_conf.c, bytes);

	float ****conv51_filter = (float ****)alloc_4D(output51_conf.c, conv51_conf.h, conv51_conf.w, input51_conf.c, bytes);
	float ****conv52_filter = (float ****)alloc_4D(output52_conf.c, conv52_conf.h, conv52_conf.w, input52_conf.c, bytes);
	float ****conv53_filter = (float ****)alloc_4D(output53_conf.c, conv53_conf.h, conv53_conf.w, input53_conf.c, bytes);

	float **fc1_filter = (float **)alloc_2D(input6_conf.h * input6_conf.w * input6_conf.c, output6_conf , bytes);
	float **fc2_filter = (float **)alloc_2D(input7_conf, output7_conf, bytes);
	float **fc3_filter = (float **)alloc_2D(input8_conf, output8_conf, bytes);
	
	//Group 1
	conv_relu_forward(input11, output11, conv11_filter, conv11_conf, input11_conf, output11_conf);
	conv_relu_forward(output11, output12, conv12_filter, conv12_conf, input12_conf, output12_conf);
	pool_forward(output12, output13, input13_conf, pool1_conf);

	// //Group 2
	conv_relu_forward(output13, output21, conv21_filter, conv21_conf, input21_conf, output21_conf);
	conv_relu_forward(output21, output22, conv22_filter, conv22_conf, input22_conf, output22_conf);
	pool_forward(output22, output23, input23_conf, pool2_conf);

	// //Group 3
	conv_relu_forward(output23, output31, conv31_filter, conv31_conf, input31_conf, output31_conf);
	conv_relu_forward(output31, output32, conv32_filter, conv32_conf, input32_conf, output32_conf);
	conv_relu_forward(output32, output33, conv33_filter, conv33_conf, input33_conf, output33_conf);
	pool_forward(output33, output34, input34_conf, pool3_conf);
	
	// //Group 4
	conv_relu_forward(output34, output41, conv41_filter, conv41_conf, input41_conf, output41_conf);
	conv_relu_forward(output41, output42, conv42_filter, conv42_conf, input42_conf, output42_conf);
	conv_relu_forward(output42, output43, conv43_filter, conv43_conf, input43_conf, output43_conf);
	pool_forward(output43, output44, input44_conf, pool4_conf);

	// //Group 5
	conv_relu_forward(output44, output51, conv51_filter, conv51_conf, input51_conf, output51_conf);
	conv_relu_forward(output51, output52, conv52_filter, conv52_conf, input52_conf, output52_conf);
	conv_relu_forward(output52, output53, conv53_filter, conv53_conf, input53_conf, output53_conf);
	pool_forward(output53, output54, input54_conf, pool5_conf);

	//fc1
	linearize_conv(output54, output6, fc1_filter, input6_conf, output6_conf);
	
	//fc2
	fc_forward(output6, output7, fc2_filter, input7_conf, output7_conf);
	
	//fc3
	fc_forward(output7, output8, fc3_filter, input8_conf, output8_conf);
}