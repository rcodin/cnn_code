#include <stdio.h>
#include <cstdlib>

using namespace std;

struct Conv_conf {
	int h;
	int w;
	int in_c;
	int out_c;
	int f_h;
	int f_w;
};

struct Pool_conf {
	int h;
	int w;
};



struct Input_conf {
	int h;
	int w;
};