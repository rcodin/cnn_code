CFLAGS = -std=c++11
INCLUDE_PATH = -I../include
FLAGS = -lcnn -fopenmp -lz
INTEL_FLAGS = -mkl
OPENCV_FLAGS = -lopencv_core -lopencv_imgproc -lopencv_highgui -lopencv_imgcodecs

example_conf2:
	g++ -I../include $(CFLAGS) example_conf2.cpp -o conf2.exec $(FLAGS)
	g++ -I../include $(CFLAGS) example_tiled_conf2_new.cpp -o conf2_tiled.exec $(FLAGS)

example_conf1:
	g++ -I../include $(CFLAGS) example_conf1.cpp -o conf1.exec $(FLAGS)
	g++ -I../include $(CFLAGS) example_tiled_conf1_new.cpp -o conf1_tiled.exec $(FLAGS)
example_im2col:
	icpc -I../include $(CFLAGS) example_im2col.cpp -o im2col.exec $(INTEL_FLAGS) $(FLAGS)
example_npy:
	g++  $(CFLAGS) example_npy.cpp -o npy.exec $(FLAGS)
image_read:
	icpc $(INCLUDE_PATH) $(CFLAGS) read_image.cpp -o read_img.exec $(FLAGS) $(OPENCV_FLAGS)