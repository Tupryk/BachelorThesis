import os
import json
import numpy as np


def arr2cstr(a):
	"""
	Converts a numpy array to a C-style string.
	"""
	return np.array2string(a,
		separator=',',
		floatmode='unique',
		threshold = 1e6,
		max_line_width = 1e6).replace('\n','').replace(' ', '').replace(',', ', ').replace('[','{ ').replace(']',' }')


def exportTree(model_path):
	gen_file = open('tree.h', 'w')
	gen_file.write(f"// GENERATED FILE FROM MODEL {model_path}\n")
	gen_file.write("#ifndef __GEN_TREE__\n")
	gen_file.write("#define __GEN_TREE__\n\n")

	data = json.load(open(model_path, "r"))
	trees_list = data["learner"]["gradient_booster"]["model"]["trees"]
	tree_data = trees_list[0]

	INPUT_SIZE = tree_data["tree_param"]["num_feature"]
	OUTPUT_SIZE = int(data["learner"]["learner_model_param"]["num_target"])
	exportname = os.path.splitext(os.path.basename(model_path))[0]

	gen_file.write(f"#define INPUT_SIZE {INPUT_SIZE}\n")
	gen_file.write(f"#define OUTPUT_SIZE {OUTPUT_SIZE}\n\n")
	gen_file.write("void tree_forward(float input[INPUT_SIZE], float output[OUTPUT_SIZE]);\n\n")
	gen_file.write(f"#endif\n")

	gen_file = open('tree.c', 'w')
	gen_file.write(f"// GENERATED FILE FROM MODEL {model_path}\n")
	gen_file.write(f'#include "tree.h"\n\n\n')

	values = ["left_children", "right_children", "split_indices", "split_conditions"]

	result = "struct DecisionTreeEnsemble\n{\n"
	for i, tree in enumerate(trees_list):
		node_count = len(tree["left_children"])
		if i != 0: result += "\n"
		for v in values:
			if v == "split_conditions":
				result += f"	float {v}_{i}[{node_count}];\n"
			else:
				result += f"	int {v}_{i}[{node_count}];\n"
	result += "};\n\n"

	result += f"struct DecisionTreeEnsemble {exportname} = {{\n"
	for i, tree in enumerate(trees_list):
		if i != 0: result += "\n"
		for v in values:
			result += f"	.{v}_{i} = {arr2cstr(np.array(tree[v]))},\n"
	result += "};\n\n"

	result += f"float tranverse_tree(int* left_children, int* right_children, int* split_indices, float* split_conditions, float input[INPUT_SIZE])\n{{\n"
	result += "	int prev_index = 0;\n"
	result += "	int index = 0;\n"
	result += "	while (index != -1)\n"
	result += "	{\n"
	result += "		prev_index = index;\n"
	result += "		if (split_conditions[index] < input[split_indices[index]]) {\n"
	result += "			index = left_children[index];\n"
	result += "		} else {\n"
	result += "			index = right_children[index];\n"
	result += "		}\n"
	result += "	}\n"
	result += "	return split_conditions[prev_index];\n"
	result += "}\n\n"

	result += "void tree_forward(float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {\n"
	for i in range(OUTPUT_SIZE):
		result += f"		output[{i}] = 0;\n"

		for j in range(i, len(trees_list), OUTPUT_SIZE):
			result += f"		output[{i}] += tranverse_tree({exportname}.left_children_{j}, {exportname}.right_children_{j}, {exportname}.split_indices_{j}, {exportname}.split_conditions_{j}, input);\n"

		result += f"		output[{i}] /= {len(trees_list)}/OUTPUT_SIZE;"
	result += "}\n"

	gen_file.write(result)


if __name__ == '__main__':
	path = "../models/forest.json"
	exportTree(path)
