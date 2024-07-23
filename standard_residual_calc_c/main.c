#include <stdio.h>

#include "basic_residual_calculator.h"
#include "test_data.h"


int main()
{
	printf("[");
	for (int i = 0; i < 100; i++)
	{
		printf("[");
		float output[OUTPUT_SIZE];
		float pwm[5] = {input[i][0], input[i][1], input[i][2], input[i][3], input[i][4]};
		float R02 = input[i][5];
		float R12 = input[i][6];
		float acc_x = input[i][7];
		float acc_y = input[i][8];
		basic_residual(output, pwm, R02, R12, acc_x, acc_y);
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
