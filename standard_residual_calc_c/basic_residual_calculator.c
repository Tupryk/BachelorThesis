#include "basic_residual_calculator.h"

#define BRUSHLESS 1
#if BRUSHLESS
    #define QUADROTOR_MASS 0.0444f
#else
    #define QUADROTOR_MASS 0.0347f
#endif

float pwm_to_force(float pwm) {
    #if BRUSHLESS
        float a = 0.0f;
        float b = 0.0005492858445116151f;
        float c = -5.360718677769569f;
    #else
        float a = 0.00000000165049399f;
        float b = 0.00009443961289999999f;
        float c = -0.37774805200000000083f;
    #endif
    return pwm*pwm*a + pwm*b + c;
}

void basic_residual(float output[OUTPUT_SIZE], float pwm[4], float R02, float R12, float w_acc_x, float w_acc_y)
{
    float u0 = 0;
    u0 += pwm_to_force(pwm[0]);
    u0 += pwm_to_force(pwm[1]);
    u0 += pwm_to_force(pwm[2]);
    u0 += pwm_to_force(pwm[3]);
    u0 *= g2N;

    output[0] = w_acc_x*QUADROTOR_MASS - R02*u0;
    output[1] = w_acc_y*QUADROTOR_MASS - R12*u0;
}
