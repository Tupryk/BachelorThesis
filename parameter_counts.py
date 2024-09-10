"""
Aproximate parameter count for our two learning models
"""

nn_parameter_count = 0
input_layer_size = 12
hidden_layer_size = 16
output_layer_size = 2
hidden_layer_count = 2

nn_parameter_count += input_layer_size
nn_parameter_count += hidden_layer_size*input_layer_size + hidden_layer_size
for _ in range(hidden_layer_count):
    nn_parameter_count += hidden_layer_size*hidden_layer_size + hidden_layer_size
nn_parameter_count += hidden_layer_size*output_layer_size + output_layer_size

print("Neural network parameter count: ", nn_parameter_count)

dte_parameter_count = 0
trees_per_class = 5
class_count = 2
parent_size = 4
parent_nodes_per_tree = 270
leaf_nodes_per_tree = 270

dte_parameter_count = class_count * (parent_nodes_per_tree*parent_size + leaf_nodes_per_tree) * trees_per_class

print("Decision tree ensemble parameter count: ", dte_parameter_count)
