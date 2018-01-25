program_NAME := libcnn.so

SRCS = blas.cpp layers_forward_pass.cpp layers_tiled_forward_pass.cpp
INCLUDE_DIR := ../include
CFLAGS = -std=c++11


.PHONY: all

all: $(program_NAME)

$(program_NAME): $(SRCS) 
	g++ -fPIC -I$(INCLUDE_DIR) $(CFLAGS) -O -c $(SRCS)
	g++ -shared -o libcnn.so blas.o layers_forward_pass.o layers_tiled_forward_pass.o -lm