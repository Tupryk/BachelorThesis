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
		for (int i = 0; i < OUTPUT_SIZE; i++)
		{
			if (i == OUTPUT_SIZE - 1)
				printf("%f", model_output[i]);
			else
				printf("%f, ", model_output[i]);
		}
		if (i == 99)
			printf("]");
		else
			printf("], ");
	}
	printf("]\n");
	return 0;
}
