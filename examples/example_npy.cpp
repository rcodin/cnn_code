#include <cnpy.hpp>
#include <vector>
#include <iostream>

using namespace std;

int main() {
	cnpy::NpyArray arr = cnpy::npy_load("/home/ronit/Videos/npy/conv1_1_W.npy");
	float *loaded_data = arr.data<float>();
	for (int i = 0; i < 3*3*3*64; i++)
		cout<<loaded_data[i]<<std::endl;
}