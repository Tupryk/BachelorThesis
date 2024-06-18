#include <stdio.h>

#include "nn.h"
#include "test_data.h"
#include "minmax_scaler.h"


int main()
{
	printf("[");
	for (int i = 0; i < 100; i++)
	{
		float scaled_output[6];
		printf("[");
		const float *model_output = nn_forward(input[i]);
		scale_output(model_output, scaled_output);
		for (int i = 0; i < OUTPUT_SIZE; i++)
		{
			if (i == OUTPUT_SIZE - 1)
				printf("%f", scaled_output[i]);
			else
				printf("%f, ", scaled_output[i]);
		}
		if (i == 99)
			printf("]");
		else
			printf("], ");
	}
	printf("]\n");
	return 0;
}
