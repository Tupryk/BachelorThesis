#ifndef __TREE_UTILS__
#define __TREE_UTILS__

float tranverse_tree(__INT16_TYPE__* left_children, __INT16_TYPE__* right_children, __UINT8_TYPE__* split_indices, float* split_conditions, float* base_weights, float* input);
float tranverse_tree_sklearn(__INT16_TYPE__* left_children, __INT16_TYPE__* right_children, __UINT8_TYPE__* split_indices, float* split_conditions, float* output_value, float* input);
float tranverse_compact_tree(__INT16_TYPE__* left_children, __INT16_TYPE__* right_children, __UINT8_TYPE__* split_indices, float* split_conditions, float* output_value, float* input);

#endif
