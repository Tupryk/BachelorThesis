import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import LMCE.cfusdlog as cfusdlog
import numpy as np
import matplotlib.pyplot as plt
from LMCE.residual_calculation import residual


data_path = "../crazyflie-data-collection/olddata/new_data/nn_log04"
data = cfusdlog.decode(data_path)['fixedFrequency']
timestamp = data["timestamp"]

f, _ = residual(data, use_rpm=False)

residuals = [np.linalg.norm(f_[:2]) for f_ in f]
plt.plot(timestamp, residuals, label="Residual")

F_d = [np.sqrt(data["lee.Fd_x"][i]**2+data["lee.Fd_y"][i]**2)*.034 for i, _ in enumerate(data["lee.Fd_x"])]
plt.plot(timestamp, F_d, label="F_d")

plt.xlabel("Time")
plt.legend()
plt.show()

diff = [np.abs(r/F_d[i])*100 for i, r in enumerate(residuals)]
print(residuals[200])
print(F_d[200])
print(diff[200])
plt.plot(timestamp, diff)
plt.title("Proportional difference")
plt.xlabel("Timestamp")
plt.ylabel("%")
plt.show()
