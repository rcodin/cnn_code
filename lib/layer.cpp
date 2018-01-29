#include <layer.hpp>

class Layer {
	private:
		std::string type;
		// (H, W, M)
		std::vector<int> size;
	public:
		Layer(std::string type, int x, int y) {
			if (type != "pool") {
				std::cerr<<"wrong Layer config"<<std::endl;
				return;
			}
			this->type = type;
			size.push_back(x);
			size.push_back(y);
		}
		Layer(std::string type, int x, int y, int z) {
			if (type != "conv_relu" || type != "conv") {
				std::cerr<<"Wrong layer config"<<std::endl;
			}
			this->type = type;
			size.push_back(x);
			size.push_back(y);
			size.push_back(z);
		}
		std::string get_type() {
			return type;
		}
		std::vector<int> get_size() {
			return size;
		}
};