CFLAGS = -std=c++11
FLAGS = -lcnn -fopenmp

example_conf2:
	g++ -I../include $(CFLAGS) example_conf2.cpp -o conf2.exec $(FLAGS)
	g++ -I../include $(CFLAGS) example_tiled_conf2_new.cpp -o conf2_tiled.exec $(FLAGS)

example_conf1:
	g++ -I../include $(CFLAGS) example_conf1.cpp -o conf1.exec $(FLAGS)
	g++ -I../include $(CFLAGS) example_tiled_conf1_new.cpp -o conf1_tiled.exec $(FLAGS)