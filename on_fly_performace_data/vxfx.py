import cfusdlog
import numpy as np
import matplotlib.pyplot as plt
from residual_calculation import residual


data_path = "./new_data/nn_log04"
data = cfusdlog.decode(data_path)['fixedFrequency']
timestamp = data["timestamp"][1:]
f, tau = residual(data)

for i, v in enumerate(["x", "y", "z"]):
    plt.scatter(np.abs(data[f"stateEstimate.v{v}"][1:]), np.abs(f[:, i]), label=f"Velocity to force in {v}")
    plt.xlabel("Velocity")
    plt.ylabel("Force")
    plt.show()

vx = data[f"stateEstimate.vx"]
vy = data[f"stateEstimate.vy"]
vz = data[f"stateEstimate.vz"]
vels = [np.sqrt(vx[i]**2 + vy[i]**2 + vz[i]**2) for i, _ in enumerate(vx)]
forces = [np.linalg.norm(f_) for f_ in f]
plt.scatter(vels[1:], forces, alpha=.1)
plt.xlabel("Velocity")
plt.ylabel("Force")
plt.show()
