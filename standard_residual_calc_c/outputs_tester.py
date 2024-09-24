import os
import sys
import subprocess
import numpy as np
from dataset_generator import data_load

sys.path.append(os.path.join(os.path.dirname(__file__), "../"))
import LMCE.cfusdlog as cfusdlog
from LMCE.residual_calculation import residual


if __name__ == '__main__':
	# Check if the function outputs match
	test_data_path = "../crazyflie-data-collection/jana_flight_data/jana02"
	data_load(test_data_path)
	print("Comparing with original...")
	process = subprocess.Popen('gcc main.c basic_residual_calculator.c -o p', shell=True)
	process.wait()
	process = subprocess.Popen('./p', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	c_output, _ = process.communicate()
	c_output = c_output.decode('utf-8')
	c_output = np.array(eval(c_output))

	data = cfusdlog.decode(test_data_path)['fixedFrequency']
	py_output, _ = residual(data, use_rpm=False, total_mass=.0444, is_brushless=True)

	same = True
	for i in range(len(c_output)):
		for j in range(2):
			print(c_output[i][j], py_output[i][j])
			if np.abs(c_output[i][j]-py_output[i][j]) >= 1e-4:
				same = False
				break
		if not same:
			break

	if same:
		print("Models give the same outputs!")
	else:
		print("Something went wrong, outputs don't match!")
