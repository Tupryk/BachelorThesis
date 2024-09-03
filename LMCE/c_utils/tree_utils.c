#include "tree_utils.h"
#include <stdio.h>


float tranverse_tree(__INT16_TYPE__* left_children, __INT16_TYPE__* right_children, __UINT8_TYPE__* split_indices, float* split_conditions, float* output_value, float* input)
{
	__INT16_TYPE__ index = 0;
	while (!(index & 0b0100000000000000))
	{
		if (split_conditions[index] >= input[split_indices[index]]) {
			index = left_children[index];
		} else {
			index = right_children[index];
		}
	}
	return output_value[index & 0b1011111111111111];
}
