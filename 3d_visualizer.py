import os
import torch
import shutil
import numpy as np
import matplotlib.pyplot as plt

import LMCE.cfusdlog as cfusdlog
from LMCE.model import MLP
from LMCE.residual_calculation import brushless_residual
from LMCE.data_prep import prepare_data, create_dataloader
from LMCE.model_to_c_conversion import exportNet, c_model_test

data_path = f"./crazyflie-data-collection/brushless_flights_payload/data/eckart{24}"
visual_scale = 1.

fig = plt.figure()
ax = plt.axes(projection='3d')

data = cfusdlog.decode(data_path)['fixedFrequency']
label = os.path.basename(data_path)

x = [i for i in data["stateEstimate.x"]]
y = [i for i in data["stateEstimate.y"]]
z = [i for i in data["stateEstimate.z"]]
origin = np.array([x, y, z]).T

f, _ = brushless_residual(data, use_rpm=True)
vector = np.array([f[:, 0], f[:, 1], f[:, 2]]).T * visual_scale
ax.quiver(origin[:, 0], origin[:, 1], origin[:, 2], vector[:, 0], vector[:, 1], vector[:, 2], color='r', alpha=.1, label="Residual with rpm scaled by 10")

ax.plot3D(x, y, z, label=label)

# Calculate the avg. residual force for each axis
for j, v in enumerate(["x", "y", "z"]):
    f_j = [f_[j] for f_ in f]
    print(f"{label} mean residual f_{v}: {sum(f_j)/len(f_j)}")

plt.title("Fx and Fy on to quadrotor trajectory")
plt.legend()
plt.axis('equal')
plt.show()
