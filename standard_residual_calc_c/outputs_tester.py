import os
import sys
import subprocess
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "../on_fly_performace_data/"))
import cfusdlog # type: ignore


if __name__ == '__main__':
	# Check if the function outputs match
	print("Comparing with original...")
	process = subprocess.Popen('gcc main.c basic_residual_calculator.c -o p', shell=True)
	process.wait()
	process = subprocess.Popen('./p', shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

	c_output, _ = process.communicate()
	c_output = c_output.decode('utf-8')
	c_output = np.array(eval(c_output))[1:]

	sys.path.append(os.path.join(os.path.dirname(__file__), "../on_fly_performace_data/"))
	from residual_calculation import residual # type: ignore
	data = cfusdlog.decode("../flight_data/jana02")['fixedFrequency']
	py_output, _ = residual(data)

	same = True
	for i in range(len(c_output)):
		for j in range(2):
			if np.abs(c_output[i][j]-py_output[i][j]) >= 1e-4:
				same = False
				break
		if not same:
			break

	if same:
		print("Models give the same outputs!")
	else:
		print("Something went wrong, outputs don't match!")
