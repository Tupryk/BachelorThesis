import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import rowan
import LMCE.cfusdlog as cfusdlog
import numpy as np
import matplotlib.pyplot as plt

data = cfusdlog.decode("../crazyflie-data-collection/olddata/payload_data/cf3_t_03")['fixedFrequency']

acc_x = []
acc_xb = []

for i in range(1, len(data['timestamp'])):

    quat = np.array([data['stateEstimate.qw'][i], data['stateEstimate.qx'][i],
                        data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]])
    R = rowan.to_matrix(quat)
    
    acc = R @ np.array([data['acc.x'][i], data['acc.y'][i], data['acc.z'][i]])
    acc[2] -= 1.
    acc *= 9.81
    acc_x.append(acc[0])
    

    acc = np.array([data['stateEstimate.ax'][i], data['stateEstimate.ay'][i], data['stateEstimate.az'][i]-1])*9.81
    acc_xb.append(acc[0])

plt.plot(acc_x, label="world")
plt.plot(acc_xb, label="quad")
plt.legend()
plt.show()
