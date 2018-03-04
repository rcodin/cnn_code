class Layer {
	public:
		float *weights;
		float *biases;
		virtual void forward();

};