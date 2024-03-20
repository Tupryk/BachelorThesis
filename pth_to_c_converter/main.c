#include <stdio.h>

#include "nn.h"


int main()
{
	float input[INPUT_SIZE] = { .0, .0, .5, .0, .0, .0, .1, 3., .5, .1, .4, 1.9 };
	const float* output = nn_forward(input);
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		printf("%f\n", output[i]);
	}
	return 0;
}
