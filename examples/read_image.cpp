#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <vector>

using namespace cv;
using namespace std;

int main()
{
    Mat mat;
    mat = imread("/home/ronit/project/benchmark/vgg_16/tensorflow/laska.png", CV_LOAD_IMAGE_COLOR);
    uchar *array;

    if(! mat.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    if (mat.isContinuous()) {
        array = mat.data;
        for (int i = 0; i < mat.rows * mat.cols; i++)
            cout<<(float)array[i]<<endl;
        // cout<<(double)array[0]<<endl;
        cout<<"Data is continuous"<<endl;
        cout<<"Size of dats is : "<<mat.rows * mat.cols<<endl;
    }

    // namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    // imshow( "Display window", image );                   // Show our image inside it.

    // waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}