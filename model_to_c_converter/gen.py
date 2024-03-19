import os
import torch
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


def exportNet(pth_path):
	gen_file = open('nn.h', 'w')
	gen_file.write(f"// GENERATED FILE FROM MODEL {path}\n")
	gen_file.write("#ifndef __GEN_NN__\n")
	gen_file.write("#define __GEN_NN__\n\n")

	state_dict = torch.load(pth_path)
	INPUT_SIZE = state_dict[list(state_dict.keys())[0]].shape[1]
	OUTPUT_SIZE = state_dict[list(state_dict.keys())[-1]].shape[0]
	gen_file.write(f"#define INPUT_SIZE {INPUT_SIZE}\n")
	gen_file.write(f"#define OUTPUT_SIZE {OUTPUT_SIZE}\n\n")
	gen_file.write("const float* nn_forward(float input[INPUT_SIZE]);\n\n")
	gen_file.write(f"#endif\n")

	gen_file = open('nn.c', 'w')
	# File info and defines
	gen_file.write(f"// GENERATED FILE FROM MODEL {path}\n")
	gen_file.write(f'#include "nn.h"\n')
	gen_file.write(f'#include "nn_utils.h"\n\n\n')
	
	exportname = os.path.splitext(os.path.basename(pth_path))[0]

	# Neural net struct
	result = "struct NeuralNetwork\n{\n"
	for key, data in state_dict.items():
		name = key.replace(".", "_")
		a = data.numpy().T
		result += "	float " + name
		for s in a.shape:
			result += "[" + str(s) + "]"
		result += ";\n"
		if len(a.shape) == 1: # Weights (Can't be sure if there is a bias but there are always weights...)
			result += f"	float {name.replace('bias', 'activation')}[{a.shape[0]}];\n"
	result += "};\n\n"

	# Weights and biases
	result += f"struct NeuralNetwork {exportname} = {{\n"
	for key, data in state_dict.items():
		name = key.replace(".", "_")
		a = data.numpy().T
		result += "	." + name + " = " + arr2cstr(a) + ",\n"
		if len(a.shape) == 1:
			result += "	." + name.replace('bias', 'activation') + " = " + arr2cstr(np.zeros(a.shape[0])) + ",\n"
	result += "};\n\n"

	result += "const float* nn_forward(float input[INPUT_SIZE]) {\n"
	switcher = 0
	counter = 0
	for key, data in state_dict.items():
		switcher += 1
		counter += 1
		input = "input" if counter == 1 else f"{exportname}.{name.replace('bias', 'activation')}"
		name = key.replace(".", "_")
		output = name.replace('bias', 'activation')
		a = data.numpy().T
		if switcher == 1:
			result += f"	layer({a.shape[0]}, {a.shape[1]}, {input}, {exportname}.{name}, "
		elif switcher == 2:
			result += f"{exportname}.{name}, {exportname}.{output}, {int(len(list(state_dict.items())) != counter)});\n"
			switcher = 0
	result += "	return " + exportname + "." + output + ";\n};\n"

	gen_file.write(result)


if __name__ == '__main__':
	path = "../models/jana_nn.pth"
	exportNet(path)
