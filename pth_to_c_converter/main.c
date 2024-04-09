#include <stdio.h>

#include "nn.h"


int main()
{
	float input[INPUT_SIZE] = { .1, -1., 3.5, -.05, -2., .04, .14, 0.1, 2.5, -.1, .4, 0.9 };
	const float* output = nn_forward(input);
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		printf("%f\n", output[i]);
	}
	return 0;
}
