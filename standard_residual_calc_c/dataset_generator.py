import os
import sys
import rowan
import numpy as np
from sklearn import preprocessing

sys.path.append(os.path.join(os.path.dirname(__file__), "../on_fly_performace_data/"))
import cfusdlog # type: ignore


def arr2cstr(a):
	"""
	Converts a numpy array to a C-style string.
	"""
	return np.array2string(a,
		separator=',',
		floatmode='unique',
		threshold = 1e6,
		max_line_width = 1e6).replace('\n','').replace(' ', '').replace(',', ', ').replace('[','{ ').replace(']',' }')

def data_load():
    data = cfusdlog.decode("../flight_data/jana02")['fixedFrequency']
    new_data = []
    count = 100
    pwm_1 = preprocessing.normalize(data['pwm.m1_pwm'][None])[0]
    pwm_2 = preprocessing.normalize(data['pwm.m2_pwm'][None])[0]
    pwm_3 = preprocessing.normalize(data['pwm.m3_pwm'][None])[0]
    pwm_4 = preprocessing.normalize(data['pwm.m4_pwm'][None])[0]
    mv = preprocessing.normalize(data['pm.vbatMV'][None])[0]
    for i in range(count):
        quat = np.array([data['stateEstimate.qw'][i], data['stateEstimate.qx'][i],
                         data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]])
        R = rowan.to_matrix(quat)
        acc = R @ np.array([data['acc.x'][i], data['acc.y'][i], data['acc.z'][i]])
        acc *= 9.81
        pwm = [data[f'pwm.m{j}_pwm'][i] for j in range(1, 5)]
        new_data.append([*pwm, data['pm.vbatMV'][i], R[0, 2], R[1, 2], acc[0], acc[1]])
        # new_data.append([pwm_1[i], pwm_2[i], pwm_3[i], pwm_4[i], mv[i], R[0, 2], R[1, 2], acc[0], acc[1]])

    gen_file = open('test_data.h', 'w')
    gen_file.write(f"const float input[{count}][9] = ")
    gen_file.write(arr2cstr(np.array(new_data)))
    gen_file.write(";\n")

    return new_data

if __name__ == '__main__':
    data_load()
    