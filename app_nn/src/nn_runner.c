#include <string.h>
#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <time.h>

#include "app.h"

#include "FreeRTOS.h"
#include "task.h"
#include "usec_time.h"

#define DEBUG 1
#include "debug.h"

#include "nn.h"

uint64_t start_time, end_time;

void appMain()
{
	DEBUG_PRINT("Running neural network...\n");

	while (1)
	{
		vTaskDelay(M2T(2000));
		start_time = usecTimestamp();

		// main loop //
        float input[INPUT_SIZE];
        for (int i = 0; i < INPUT_SIZE; i++) {
            input[i] = ((float)rand() / RAND_MAX) * 2 - 1;
        }
        const float *output = nn_forward(input);
		// --------- //

        end_time = usecTimestamp();

		#if DEBUG
		DEBUG_PRINT("==================================================\n");
		double elapsed_time = end_time-start_time;
		double hz = 1000.0/elapsed_time;
		// This might be wrong... Im not sure if this is in milliseconds
		DEBUG_PRINT("Neural network took %f milliseconds to output result. (%fHz)\n", elapsed_time, hz);
		DEBUG_PRINT("INPUT: { ");
		for (int i = 0; i < INPUT_SIZE; i++) {
			if (i != INPUT_SIZE-1) {
				DEBUG_PRINT("%f, ", (double)input[i]);
			} else {
				DEBUG_PRINT("%f", (double)input[i]);
			}
		}
		DEBUG_PRINT(" }\nOUTPUT: { ");
		for (int i = 0; i < OUTPUT_SIZE; i++) {
			if (i != OUTPUT_SIZE-1) {
				DEBUG_PRINT("%f, ", (double)output[i]);
			} else {
				DEBUG_PRINT("%f", (double)output[i]);
			}
		}
		DEBUG_PRINT(" }\n");
		#endif
	}
}
