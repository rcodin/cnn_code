program_NAME := libcnn.so

SRCS = blas.cpp layers_forward_pass.cpp layers_tiled_forward_pass.cpp layer.cpp net.cpp
OBJS = blas.o layers_forward_pass.o layers_tiled_forward_pass.o layer.o net.o
INCLUDE_DIR := ../include
CFLAGS = -std=c++11


.PHONY: all

all: $(program_NAME)

$(program_NAME): $(SRCS) 
	g++ -fPIC -I$(INCLUDE_DIR) $(CFLAGS) -O -c $(SRCS) -fopenmp
	g++ -shared -o libcnn.so $(OBJS) -lm -fopenmp