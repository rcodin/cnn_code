//opencv headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <input.hpp>

int read_image_rgb(std::string filename, image_cfg cfg) {
	Mat img, mat = Mat::zeros(cfg.rows, cfg.cols, CV_8UC3);

    img = imread(filename, CV_LOAD_IMAGE_COLOR);
    
    uchar *array;

    if(! img.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    resize(img, mat, Size(224, 224), 0, 0, INTER_LINEAR);

    if (mat.isContinuous()) {
        array = mat.data;
        
        
    }

}