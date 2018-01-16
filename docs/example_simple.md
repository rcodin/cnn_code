			Arbitrary Network Configuration to start with 
Input :

224x224x3

Conv1:
	3x3x3x64
Pool1:
	3x3
Conv2:
	3x3x64x128
Pool2:
	3x3
Conv3:
	3x3x128x256
Pool3:
	2x2
Fc1:
	(11x11x256)x512
Fc2:
	256x256
Softmax_loss:
	256x1