#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

// -i input
// -w weights
int read_config(int argc, char** argv, string &weight_file, string &image_file) {
    
    if (argc != 5) {
        cerr<<"Error: invalid options"<<endl;
        return -1;
    }
    for (int i = 1; i < argc; i++) {
        if (argv[i] == '-i') {
            image_file = argv[i + 1];
            i++;
        }
        else if (argv[i] == '-w') {
            weight_file = argv[i + 1];
            i++;
        }
        else {
            return -1;
        }
    }
    return 0;
}

int main(int argc, char** argv)
{
    string weight_file;
    string image_file;

    int err = read_config(argc, argv, weight_file, image_file);

    if (err) {
        return -1;
    }

    vector<float> input = 
}