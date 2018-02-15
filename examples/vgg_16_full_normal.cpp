#include <stdio.h>
#include <cstdlib>
#include <stdint.h>
#include <layers.hpp>
#include <tiling.hpp>
#include <utils.hpp>

using namespace std;

int main() {
	//conv->relu->pool conv->relu->pool conv->relu->pool fc->fc

	//load network weights
	size_t bytes = sizeof(float);
	
	//create input 
	//conv1->relu->pool
	{
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
		Conv_conf conv11_conf = {3, 3};	
		Data_conf input11_conf = {112, 112, 64};
		Data_conf output11_conf = {112, 112, 128};

		//Conv22
		Conv_conf conv22_conf = {3, 3};
		Data_conf input11_conf = {112, 112, 128};
		Data_conf output11_conf = {112, 112, 128};

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
		Data_conf input33_conf = {56, 56, 256};
		Data_conf output33_conf = {28, 28, 256};

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
		int input8_conf = 1000;
	}

	float ****conv11_filter = (float ****)alloc_4D(64, 3, 3, 3, bytes);
	float ****conv12_filter = (float ****)alloc_4D(128, 3, 3, 64, bytes);


	float ***input11 = (float ***)alloc_3D(input11_tiled_conf.h, input11_tiled_conf.w, input11_conf.c, bytes);
	float ***output11 = (float ***)alloc_3D(output11_tiled_conf.h, output11_tiled_conf.w, output11_conf.c, bytes);
	float ***output12 = (float ***)alloc_3D(output12_tiled_conf.h, output12_tiled_conf.w, output12_tiled_conf.c, bytes);
	float ***output13 = (float ***)alloc_3D(output13_tiled_conf.h, output13_tiled_conf.w, output13_tiled_conf.c, bytes);

	float ***output21 = (float ***)alloc_3D(output21_tiled_conf.h, output21_tiled_conf.w, output21_conf.c, bytes);
	float ***output22 = (float ***)alloc_3D(output22_tiled_conf.h, output22_tiled_conf.w, output22_tiled_conf.c, bytes);
	float ***output23 = (float ***)alloc_3D(output23_tiled_conf.h, output23_tiled_conf.w, output23_tiled_conf.c, bytes);

	float ***output31 = (float ***)alloc_3D(output31_tiled_conf.h, output31_tiled_conf.w, output31_conf.c, bytes);
	float ***output32 = (float ***)alloc_3D(output32_tiled_conf.h, output32_tiled_conf.w, output32_tiled_conf.c, bytes);
	float ***output33 = (float ***)alloc_3D(output33_tiled_conf.h, output33_tiled_conf.w, output33_tiled_conf.c, bytes);

	float ***output41 = (float ***)alloc_3D(output41_tiled_conf.h, output41_tiled_conf.w, output41_conf.c, bytes);
	float ***output42 = (float ***)alloc_3D(output42_tiled_conf.h, output42_tiled_conf.w, output42_tiled_conf.c, bytes);
	float ***output43 = (float ***)alloc_3D(output43_tiled_conf.h, output43_tiled_conf.w, output43_tiled_conf.c, bytes);
	float ***output44 = (float ***)alloc_3D(output44_tiled_conf.h, output44_tiled_conf.w, output44_tiled_conf.c, bytes);

	float ***output51 = (float ***)alloc_3D(output51_tiled_conf.h, output51_tiled_conf.w, output51_conf.c, bytes);
	float ***output52 = (float ***)alloc_3D(output52_tiled_conf.h, output52_tiled_conf.w, output52_tiled_conf.c, bytes);
	float ***output53 = (float ***)alloc_3D(output53_tiled_conf.h, output53_tiled_conf.w, output53_tiled_conf.c, bytes);
	float ***output54 = (float ***)alloc_3D(output54_tiled_conf.h, output54_tiled_conf.w, output54_tiled_conf.c, bytes);

	float *output6 = (float *)alloc_1D(output6_tiled_conf, bytes);
	float *output7 = (float *)alloc_1D(output7_tiled_conf, bytes);
	float *output8 = (float *)alloc_1D(output8_tiled_conf, bytes);

	conv_relu_forward(input11, output11, conv11_filter, conv11_conf, input11_conf, input11_tiled_conf, output11_tiled_conf, input11_tile_base, output11_tile_base);
	conv_relu_forward(output11, output12, conv12_filter, conv12_conf, input12_conf, input12_tiled_conf, output12_tiled_conf, input12_tile_base, input12_tile_base);
}