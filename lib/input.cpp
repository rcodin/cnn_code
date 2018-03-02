//opencv headers
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <vector>
#include <string>
#include <input.hpp>
#include <iostream>

using namespace cv;

//read and put the image in data
int read_image_rgb(std::string filename, Image_cfg cfg, std::vector<float> &data) {
	Mat img, mat = Mat::zeros(cfg.rows, cfg.cols, CV_8UC3);

    img = imread(filename, CV_LOAD_IMAGE_COLOR);
    std::cout<<"dsds"<<std::endl;
    while(1);
    uchar *array;

    if(! img.data )                              // Check for invalid input
    {
        std::cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    resize(img, mat, Size(cfg.rows, cfg.cols), 0, 0, INTER_LINEAR);

	if (mat.isContinuous()) {
		data.assign((float*)mat.datastart, (float*)mat.dataend);
	}
	else {
		for (int i = 0; i < mat.rows; ++i) {
			data.insert(data.end(), mat.ptr<float>(i), mat.ptr<float>(i)+mat.cols);
		}
	}
	for (int i = 0; i < data.size(); i++) {
		std::cout<<data[i]<<std::endl;
	}
	std::cout<<data.size()<<std::endl;

	return 0;
}