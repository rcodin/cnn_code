program_NAME := libcnn.so

SRCS = blas.cpp layers_forward_pass.cpp layers_tiled_forward_pass.cpp layer.cpp net.cpp utils.cpp im2col.cpp cnpy.cpp input.cpp
OBJS = blas.o layers_forward_pass.o layers_tiled_forward_pass.o layer.o net.o utils.o im2col.o cnpy.o input.o
INCLUDE_DIR := ../include
CFLAGS = -std=c++11
MKL_FLAGS = -mkl 
OPENCV_FLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

.PHONY: all

all: $(program_NAME)

$(program_NAME): $(SRCS) 
	g++ -fPIC -I$(INCLUDE_DIR) $(CFLAGS) -O -c -D TILE=0 $(SRCS) -fopenmp 
	g++ -shared -o libcnn.so $(OBJS) -lm -fopenmp
TILE:
	g++ -fPIC -I$(INCLUDE_DIR) $(CFLAGS) -O -c -D TILE=1 $(SRCS) -fopenmp
	g++ -shared -o libcnn.so $(OBJS) -lm -fopenmp
MKL:
	icpc -fPIC -I$(INCLUDE_DIR) $(CFLAGS) -O -c -D TILE=1 $(SRCS) -fopenmp $(MKL_FLAGS) $(OPENCV_FLAGS)
	icpc -shared -o libcnn.so $(OBJS) -lm -fopenmp  $(MKL_FLAGS) $(OPENCV_FLAGS)