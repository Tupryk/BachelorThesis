import os
import json
import subprocess
import numpy as np
import xgboost as xg


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
	gen_file.write(f'#include "tree.h"\n')
	gen_file.write(f'#include "tree_utils.h"\n\n\n')

	values = ["left_children", "right_children", "split_indices", "split_conditions", "base_weights"]

	result = "struct DecisionTreeEnsemble\n{\n"
	for i, tree in enumerate(trees_list):
		node_count = len(tree["left_children"])
		if i != 0: result += "\n"
		for v in values:
			if v == "split_conditions" or v == "base_weights":
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

	result += "void tree_forward(float input[INPUT_SIZE], float output[OUTPUT_SIZE]) {\n"
	for i in range(OUTPUT_SIZE):
		result += f"	output[{i}] = 0;\n"

		for j in range(i, len(trees_list), OUTPUT_SIZE):
			result += f"	output[{i}] += tranverse_tree("
			for v in values:
				result += f"{exportname}.{v}_{j}, "
			result += "input);\n"

		result += f"	output[{i}] /= {len(trees_list)}/OUTPUT_SIZE;\n"
	result += "}\n"

	gen_file.write(result)


if __name__ == '__main__':
	path = "../new_model_gen/tree.json"
	exportTree(path)

	# Check if the model outputs match
	print("Comparing with original...")
	process = subprocess.Popen('gcc main.c tree.c tree_utils.c -o p', shell=True)
	process.wait()
	process = subprocess.Popen('./p', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	c_output, _ = process.communicate()
	c_output = c_output.decode('utf-8')
	c_output = np.array(eval(c_output))

	xg_model = xg.XGBRegressor()
	xg_model.load_model('../new_model_gen/tree.json')
	test_data = np.load(f"../pth_to_c_converter/test_data.npz")["array"]
	py_output = xg_model.predict(test_data)
	print(c_output[0])
	print(py_output[0])

	same = True
	for i in range(len(c_output)):
		for j in range(6):
			if np.abs(c_output[i][j]-py_output[i][j]) >= 1e-4:
				same = False
				break
		if not same:
			break

	if same:
		print("Models give the same outputs!")
	else:
		print("Something went wrong, outputs dont match!")

