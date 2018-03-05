#ifndef UTILS_HPP
#define UTILS_HPP

#include <cstdlib>
#include <iostream>
#include <layers.hpp>
#include <string>
#include <vector>

//memory manager
void *alloc_1D(int i, size_t bytes);
void *alloc_2D(int i, int j, size_t bytes);
void *alloc_3D(int i, int j, int k, size_t bytes);
void *alloc_4D(int i, int j, int k, int l, size_t bytes);
void free_mem(void *ptr);

void print_conf_cfg(Conv_conf cfg, Data_conf input_cfg, Data_conf output_cfg);
void replicate_across_cols(float *input, float *output, int rows, int cols);
int read_config(int argc, char** argv, std::string &weight_file, std::string &image_file);

#endif