#include <input.hpp>

class Driver {
	private:
		Net net;
		Input3D input;
	public:
		Driver(Net network, Input3D input) {
			this->net = network;
			this->input = input;
		}
		void run() {
			int num_layers = net.get_num_layers();

		}
};