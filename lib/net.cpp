#include <net.hpp>

class Net {
	private:
		std::vector<Layer> layers;
	public:
		Net() {
		}
		// void append_layer(Layer layer) {
		// 	layers.push_back(layer);
		// }
		// Layer access_layer(int i) {
		// 	if (layers.size() <= i) {
		// 		std::cerr<<"layer index out of boundary"<<std::endl;
		// 		return NULL;
		// 	}
		// 	return layers[i];
		// }
		int get_num_layers() {
			// return layers.size();
		}
};