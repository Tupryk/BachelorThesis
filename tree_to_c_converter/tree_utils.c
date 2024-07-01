#include "tree_utils.h"


float tranverse_tree(int* left_children, int* right_children, int* split_indices, float* split_conditions, float* base_weights, float* input)
{
	int prev_index = 0;
	int index = 0;
	while (index != -1)
	{
		prev_index = index;
		if (split_conditions[index] < input[split_indices[index]]) {
			index = left_children[index];
		} else {
			index = right_children[index];
		}
	}
	return base_weights[prev_index];
}

float tranverse_tree_sklearn(int* left_children, int* right_children, int* split_indices, float* split_conditions, float* output_value, float* input)
{
	int prev_index = 0;
	int index = 0;
	while (index >= 0)
	{
		prev_index = index;
		if (split_conditions[index] >= input[split_indices[index]]) {
			index = left_children[index];
		} else {
			index = right_children[index];
		}
	}
	return output_value[prev_index];
}

