#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>


#include <input.hpp>

using namespace cv;
using namespace std;

// -i input
// -w weights
int read_config(int argc, char** argv, string &weight_file, string &image_file) {
    
    if (argc != 5) {
        cerr<<"Error: invalid options"<<endl;
        return -1;
    }
    for (int i = 1; i < argc; i += 2) {
        // cout<<"mana"<<argv[i]<<"mana"<<std::endl;
        if (argv[i] == "-i") {
            image_file = argv[i + 1];
            // i++;
        }
        else if (argv[i] == "-w") {
            weight_file = argv[i + 1];
            // i++;
        }
        else {
            // cout<<argv[i]<<std::endl;
            return -1;
        }
    }
    return 0;
}

int main(int argc, char** argv)
{
    int err;
    string weight_file;
    string image_file;

    err = read_config(argc, argv, weight_file, image_file);

    if (err) {
        return -1;
    }

    vector<float> input;

    Image_cfg input_cfg = {224, 224};

    err = read_image_rgb(image_file, input_cfg, input);

    if (err) {
        return -1;
    }
    
}