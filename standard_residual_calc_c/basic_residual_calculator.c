#include "basic_residual_calculator.h"

float pwm_to_force(float pwm, float mv) {
    return 11.09f - 39.08f*pwm - 9.53f*mv + 20.57f*pwm*pwm + 38.43f*pwm*mv;
}

void basic_residual(float output[OUTPUT_SIZE], float pwm[5], float R02, float R12, float w_acc_x, float w_acc_y)
{
    float pwm_1 = pwm[0] / 3169399.926991f;
    float pwm_2 = pwm[1] / 3049107.418296f;
    float pwm_3 = pwm[2] / 3261586.709328f;
    float pwm_4 = pwm[3] / 3075472.064296f;
    float mv    = pwm[4] /  211742.686105f;

    float u0 = 0;
    u0 += pwm_to_force(pwm_1, mv);
    u0 += pwm_to_force(pwm_2, mv);
    u0 += pwm_to_force(pwm_3, mv);
    u0 += pwm_to_force(pwm_4, mv);
    u0 *= g2N;

    output[0] = w_acc_x*QUADROTOR_MASS - R02*u0;
    output[1] = w_acc_y*QUADROTOR_MASS - R12*u0;
}
