#include <stdio.h>

#include "tree.h"
#include "test_data.h"


int main()
{
	printf("[");
	for (int i = 0; i < 100; i++)
	{
		printf("[");
		float output[OUTPUT_SIZE];
		tree_forward(input[i], output);
		for (int j = 0; j < OUTPUT_SIZE; j++)
		{
			if (j == OUTPUT_SIZE - 1)
				printf("%f", output[j]);
			else
				printf("%f, ", output[j]);
		}
		if (i == 99)
			printf("]");
		else
			printf("], ");
	}
	printf("]\n");
	return 0;
}
