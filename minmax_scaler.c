#include "minmax_scaler.h"


const float mins[6] = { -0.2275051, -0.2195530, -0.1459336, -0.0001424, -0.0001197, -0.0000383 };
const float maxs[6] = { 0.2136143, 0.2050502, 0.0451654, 0.0001219, 0.0001645, 0.0000351 };

void scale_output(const float* model_output, float* scaled_output)
{
    for (int i = 0; i < 6; i++) {
        scaled_output[i] = (model_output[i] + 1.f) * .5f * (maxs[i] - mins[i]) + mins[i];
    }
}
