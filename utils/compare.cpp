#include <iostream>
#include <cstdlib>
#include <string>
#include <fstream>
#include <stdio.h>
#include <iomanip>

using namespace std;

int main(int argc, char **argv)
{
	string f1_line, f2_line;
	float f1_num, f2_num;

	ifstream file1 (argv[1]);
	ifstream file2 (argv[2]);

	double avg_diff = 0.0f;
	int count = 0;
	while (1) {
		if (getline(file1, f1_line)) {
			if (getline(file2, f2_line)) {
				count++;

				f1_num = stof(f1_line);
				f2_num = stof(f2_line);
				if ((f1_num - f2_num) > 3){
					cout<<count<<" "<<f1_num<<" "<<f2_num<<" "<<(f1_num - f2_num)<<endl;
				}
				avg_diff += (f1_num - f2_num);
				// std::cout<<f1_num<<" "<<f2_num<<std::endl;
			}
		}
		else {
			break;
		}
	}
	cout<<fixed<<setprecision(20)<<avg_diff/count<<endl;
}