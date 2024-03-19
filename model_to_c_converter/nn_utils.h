#ifndef __NN_H__
#define __NN_H__

void layer(int rows, int cols, float in[rows], float layer_weight[rows][cols], float layer_bias[cols], float output[cols], int use_activation);

#endif