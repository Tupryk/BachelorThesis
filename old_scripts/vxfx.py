import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import LMCE.cfusdlog as cfusdlog
import numpy as np
import matplotlib.pyplot as plt
from LMCE.residual_calculation import residual


data_path = "../crazyflie-data-collection/olddata/new_data/nn_log04"
data = cfusdlog.decode(data_path)['fixedFrequency']
timestamp = data["timestamp"][1:]
f, tau = residual(data, use_rpm=False)
f = f[1:]

for i, v in enumerate(["x", "y", "z"]):
    plt.scatter(np.abs(data[f"stateEstimate.v{v}"][1:]), np.abs(f[:, i]), c=timestamp, cmap='viridis', label=f"Velocity to force in {v}")
    plt.xlabel("Velocity")
    plt.ylabel("Force")
    plt.show()

vx = data[f"stateEstimate.vx"]
vy = data[f"stateEstimate.vy"]
vz = data[f"stateEstimate.vz"]
vels = [np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2) for i, _ in enumerate(vx)]
forces = [np.linalg.norm(f_) for f_ in f]
plt.scatter(vels[1:], forces, c=timestamp, cmap='viridis', alpha=.1)
plt.xlabel("Velocity")
plt.ylabel("Force")
plt.show()
