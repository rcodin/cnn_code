#include <stdio.h>
#include <cstdlib>
#include <istream>

using namespace std;

int set_val(int *arr) {
	for (int i = 0; i < 5; i++)
		arr[i] = i + 1;
}

int main() {
	int arr[5] = {0};
	set_val(arr);
	for (int i = 0; i < 5; i++)
		printf("%d ", arr[i]);
}