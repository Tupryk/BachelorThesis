import os
import numpy as np
import matplotlib.pyplot as plt

import LMCE.cfusdlog as cfusdlog
from LMCE.residual_calculation import residual


residual_func = lambda data: residual(data, is_brushless=False, has_payload=False, use_rpm=False, total_mass=.034)
data_path = "./timescale7/cf21/nn/nn_log00"
visual_scale = 1.

fig = plt.figure()
ax = plt.axes(projection='3d')

data = cfusdlog.decode(data_path)['fixedFrequency']
label = os.path.basename(data_path)

x = [i for i in data["stateEstimate.x"]]
y = [i for i in data["stateEstimate.y"]]
z = [i for i in data["stateEstimate.z"]]
origin = np.array([x, y, z]).T

f, _ = residual_func(data)
vector = np.array([f[:, 0], f[:, 1], f[:, 2]]).T * visual_scale
ax.quiver(origin[:, 0], origin[:, 1], origin[:, 2], vector[:, 0], vector[:, 1],
          vector[:, 2], color='r', alpha=.1, label=f"Residual forces (scaled by {visual_scale})")

ax.plot3D(x, y, z, label="Quadrotor trajectory")

# Calculate the avg. residual force for each axis
for j, v in enumerate(["x", "y", "z"]):
    f_j = [f_[j] for f_ in f]
    print(f"{label} mean residual f_{v}: {sum(f_j)/len(f_j)}")

plt.title("Residual forces on quadrotor trajectory")
plt.legend()
plt.axis('equal')
plt.show()