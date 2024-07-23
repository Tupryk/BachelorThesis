#ifndef BASIC_RESIDUAL_CALCULATOR
#define BASIC_RESIDUAL_CALCULATOR

#define OUTPUT_SIZE 2
#define g2N 0.00981f
#define QUADROTOR_MASS 0.0347f


void basic_residual(float output[OUTPUT_SIZE], float pwm[5], float R02, float R12, float w_acc_x, float w_acc_y);

#endif
