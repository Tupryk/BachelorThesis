#include "minmax_scaler.h"


const float mins[6] = { -0.2275051, -0.2195530, -0.1459336, -0.0001424, -0.0001197, -0.0000383 };
const float maxs[6] = { 0.2136143, 0.2050502, 0.0451654, 0.0001219, 0.0001645, 0.0000351 };

const float* scale_output(float output[6])
{
    for (int i = 0; i < 6; i++) {
        output[i] = (output[i] + 1) * .5 * (maxs[i] - mins[i]) + mins[i];
    }
}
