import rowan
import numpy as np
import matplotlib.pyplot as plt

import LMCE.cfusdlog as cfusdlog

data_path = "./timescale7/cf21/standard/nn_log09"

data = cfusdlog.decode(data_path)['fixedFrequency']

off_board = []
on_board = []

for i in range(100):
    quat = np.array([data['stateEstimate.qw'][i], data['stateEstimate.qx'][i],
                    data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]])
    R = rowan.to_matrix(quat)
    acc = R @ np.array([data['acc.x'][i], data['acc.y'][i], data['acc.z'][i]])
    acc *= 9.81
    pwm = [data[f'pwm.m{j}_pwm'][i] for j in range(1, 5)]
    off_board.append([*pwm, R[0, 2], R[1, 2], acc[0], acc[1]])
    on_board.append([
                    data["nn_input.se_acc_x"][i],
                    data["nn_input.se_acc_y"][i],
                    data["nn_input.se_acc_z"][i],
                    data["nn_input.gyro_x"][i],
                    data["nn_input.gyro_y"][i],
                    data["nn_input.gyro_z"][i],
                    data["nn_input.se_r_0"][i],
                    data["nn_input.se_r_1"][i]
                ])

off_board = np.array(off_board).T
on_board = np.array(on_board).T

thingies = ["pwm1", "pwm2", "pwm3", "pwm4", "R01", "R12", "acc_x", "acc_y"]
for i in range(len(off_board)):
    plt.title(thingies[i])
    plt.plot(off_board[i], label="off_board", alpha=.2)
    plt.plot(on_board[i], label="on_board", alpha=.2)
    plt.legend()
    plt.show()