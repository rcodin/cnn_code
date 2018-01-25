#include <stdio.h>

struct tile_idx_conf {
	int h_base_idx;
	int w_base_idx;
	int c_base_idx;
};

void conv_forward_tiled(float ***in, float ***out, float ****filter, Conv_conf conv_conf, 
					Data_conf input_conf, Data_conf output_conf, tile_idx_conf input_tile_conf, tile_idx_conf ouptut_tile_conf);

void pool_forward_tiled(float ***in, float ***out, Data_conf input_conf, Pool_conf pool_conf,
					tile_idx_conf input_tile_conf, tile_idx_conf output_tile_conf);

void relu_forward_tiled(float ***in, float ***out, Data_conf input_conf,
					tile_idx_conf input_tile_conf, tile_idx_conf output_tile_conf);