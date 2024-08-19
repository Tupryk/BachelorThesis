import os
import torch
import subprocess
import numpy as np


PATH = "./LMCE/c_utils/"


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
	gen_file = open(f'{PATH}nn.h', 'w')
	gen_file.write(f"// GENERATED FILE FROM MODEL {pth_path}\n")
	gen_file.write("#ifndef __GEN_NN__\n")
	gen_file.write("#define __GEN_NN__\n\n")

	state_dict = torch.load(pth_path)
	INPUT_SIZE = state_dict[list(state_dict.keys())[0]].shape[1]
	OUTPUT_SIZE = state_dict[list(state_dict.keys())[-1]].shape[0]
	gen_file.write(f"#define INPUT_SIZE {INPUT_SIZE}\n")
	gen_file.write(f"#define OUTPUT_SIZE {OUTPUT_SIZE}\n\n")
	gen_file.write("const float* nn_forward(float input[INPUT_SIZE]);\n\n")
	gen_file.write(f"#endif\n")

	gen_file = open(f'{PATH}nn.c', 'w')
	# File info and defines
	gen_file.write(f"// GENERATED FILE FROM MODEL {pth_path}\n")
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


def c_model_test(original_model, test_data_path):
	# Check if the model outputs match
	print("Comparing with original...")
	test_data = np.load(test_data_path)["X"]
	
	gen_file = open(f'{PATH}test_data.h', 'w')
	gen_file.write(f"float input[100][12] = {arr2cstr(test_data[:100])};")
	
	process = subprocess.Popen(f'gcc {PATH}main.c {PATH}nn.c {PATH}nn_utils.c -o {PATH}p', shell=True)
	process.wait()
	process = subprocess.Popen(f'{PATH}p', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
	process.wait()

	c_output, _ = process.communicate()
	c_output = c_output.decode('utf-8')
	c_output = np.array(eval(c_output))

	tensor_input = torch.from_numpy(test_data)
	py_output = original_model.forward(tensor_input).detach().numpy()

	same = True
	output_size = len(c_output[0])
	for i in range(len(c_output)):
		for j in range(output_size):
			if np.abs(c_output[i][j]-py_output[i][j]) >= 1e-4:
				same = False
				break
		if not same:
			break

	if same:
		print("Models give the same outputs!")
	else:
		print("Something went wrong, outputs dont match!")
	