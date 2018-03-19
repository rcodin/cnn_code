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
    	int err;
    
    string weight_file;
    string image_file;

    err = read_config(argc, argv, weight_file, image_file);

    if (err) {
        return -1;
    }

    float *input;

    Image_cfg input_cfg = {224, 224};
    input = (float *)malloc(input_cfg.rows * input_cfg.cols * 3);

    err = read_image_rgb(weight_file, input_cfg, input);
    if (err) {
        return -1;
    }

    size_t bytes = sizeof(float);
    int alignment = bytes * 8;
    

    

}