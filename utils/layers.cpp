#include <vector>
#include <layers.hpp>

class Layer {
	private:
		enum Layer_type type;
		int num_dims;
		std::vector<int> dimentions;
	public:
		Layer(std::string str , int num_dims, std::vector<int> dims) {
			
			if (str == "conv")
				type = conv;
			else if (str == "pool")
				type = pool;
			else if (str == "relu")
				type = relu;
			else if (str == "fc")
				type = fc;
			else
				type = no_imp;
			this->num_dims = num_dims;
			dimentions = dims;
		}
};