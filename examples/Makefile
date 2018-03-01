CFLAGS = -std=c++11
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
	icpc -I../include $(CFLAGS) example_im2col.cpp -o im2col.exec $(FLAGS)
example_npy:
	g++ -I../include $(CFLAGS) example_npy.cpp -o npy.exec $(FLAGS)
image_read:
	g++ $(CFLAGS) read_image.cpp -o read_img.exec  $(OPENCV_FLAGS)