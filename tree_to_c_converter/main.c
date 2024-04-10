#include <stdio.h>

#include "tree.h"


int main()
{
	float input[INPUT_SIZE] = { .1 };
	float output[OUTPUT_SIZE];
	tree_forward(input, output);
	for (int i = 0; i < OUTPUT_SIZE; i++) {
		printf("%f\n", output[i]);
	}
	return 0;
}
