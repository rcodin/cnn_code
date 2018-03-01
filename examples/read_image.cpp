#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <iostream>
#include <vector>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

int main()
{
    Mat img, mat = Mat::zeros(224,224, CV_8UC3);

    img = imread("/home/ronit/project/benchmark/vgg_16/tensorflow/laska.png", CV_LOAD_IMAGE_COLOR);
    
    uchar *array;

    if(! img.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    resize(img, mat, Size(224, 224), 0, 0, INTER_LINEAR);

    if (mat.isContinuous()) {
        array = mat.data;
        for (int i = 0; i < mat.rows * mat.cols * 3 + 1; i++) {
            cout<<(float)array[i]<<endl;
        }
        cout<<"Data is continuous"<<endl;
        cout<<"mat dimention: "<<img.step<<endl;
        cout<<"Size of dats is : "<<mat.rows * mat.cols<<endl;
    }


    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    // imshow( "Display window", mat);                   // Show our image inside it.

    // waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}