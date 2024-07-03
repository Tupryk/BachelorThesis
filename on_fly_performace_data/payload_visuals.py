import cfusdlog
import numpy as np
import matplotlib.pyplot as plt
from residual_calculation import payload_residual, project, project_onto_plane


fig = plt.figure()
ax = plt.axes(projection ='3d')

data_path = "./payload_data/cf3_t_04"

data = cfusdlog.decode(data_path)['fixedFrequency']
for key in data.keys():
    print(key)

x = [i*.001 for i in data["stateEstimateZ.x"]]
y = [i*.001 for i in data["stateEstimateZ.y"]]
z = [i*.001 for i in data["stateEstimateZ.z"]]

ax.plot3D(x, y, z)
f = payload_residual(data)

### Fa
origin = np.array([x, y, z]).T[1:]
vector = np.array([f[:, 0], f[:, 1], f[:, 2]]).T
ax.quiver(
    origin[:, 0], origin[:, 1], origin[:, 2],
    vector[:, 0], vector[:, 1], vector[:, 2],
    color='r', alpha=0.1, label="Fa"
)

### Tq
origin = np.array([x, y, z]).T[1:]
payload_dir = np.array([data["stateEstimateZ.px"]-data["stateEstimateZ.x"],
                        data["stateEstimateZ.py"]-data["stateEstimateZ.y"],
                        data["stateEstimateZ.pz"]-data["stateEstimateZ.z"]], dtype=np.float32).T

vector = project(f, payload_dir[1:])
ax.quiver(
    origin[:, 0], origin[:, 1], origin[:, 2],
    vector[:, 0], vector[:, 1], vector[:, 2],
    color='b', alpha=0.1, label="Tq"
)

### Fa tilde
origin = np.array([x, y, z]).T[1:]

vector = project_onto_plane(f, payload_dir[1:])
ax.quiver(
    origin[:, 0], origin[:, 1], origin[:, 2],
    vector[:, 0], vector[:, 1], vector[:, 2],
    color='g', alpha=0.1, label="Fa tilde"
)

ax.axis('equal')
ax.legend()
plt.show()
