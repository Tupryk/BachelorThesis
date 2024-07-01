#ifndef __TREE_UTILS__
#define __TREE_UTILS__

float tranverse_tree(int* left_children, int* right_children, int* split_indices, float* split_conditions, float* base_weights, float* input);
float tranverse_tree_sklearn(int* left_children, int* right_children, int* split_indices, float* split_conditions, float* output_value, float* input);

#endif
