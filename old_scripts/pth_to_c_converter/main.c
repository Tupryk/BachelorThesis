#include <stdio.h>

#include "nn.h"
#include "test_data.h"


int main()
{
	printf("[");
	for (int i = 0; i < 100; i++)
	{
		printf("[");
		const float *model_output = nn_forward(input[i]);
		for (int j = 0; j < OUTPUT_SIZE; j++)
		{
			if (j == OUTPUT_SIZE - 1)
				printf("%f", model_output[j]);
			else
				printf("%f, ", model_output[j]);
		}
		if (i == 99)
			printf("]");
		else
			printf("], ");
	}
	printf("]\n");
	return 0;
}
