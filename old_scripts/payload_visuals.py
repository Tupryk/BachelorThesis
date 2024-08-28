import cfusdlog
import numpy as np
import matplotlib.pyplot as plt
from residual_calculation import payload_residual, project, project_onto_plane

THREE_D = False

def plot_vectors(origin: np.ndarray, vector: np.ndarray, label: str="", color: str="b"):
    if THREE_D:
        ax.quiver(
            origin[:, 0], origin[:, 1], origin[:, 2],
            vector[:, 0], vector[:, 1], vector[:, 2],
            color=color, alpha=0.1, label=label
        )
    else:
        plt.quiver(
            origin[:, 0], origin[:, 1],
            vector[:, 0], vector[:, 1],
            color=color, alpha=0.1, label=label
        )

if THREE_D:
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

data_path = "./payload_data/cf3_t_04"

data = cfusdlog.decode(data_path)['fixedFrequency']
for key in data.keys():
    print(key)

x = [i*.001 for i in data["stateEstimateZ.x"]]
y = [i*.001 for i in data["stateEstimateZ.y"]]
z = [i*.001 for i in data["stateEstimateZ.z"]]

if THREE_D:
    ax.plot3D(x, y, z)
else:
    plt.plot(x, y)

f = payload_residual(data)
origin = np.array([x, y, z]).T[1:]
payload_dir = np.array([data["stateEstimateZ.px"]-data["stateEstimateZ.x"],
                        data["stateEstimateZ.py"]-data["stateEstimateZ.y"],
                        data["stateEstimateZ.pz"]-data["stateEstimateZ.z"]], dtype=np.float32).T

### Fa
vector = np.array([f[:, 0], f[:, 1], f[:, 2]]).T
plot_vectors(origin, vector, label="Fa", color="r")

### Tq
vector = project(f, payload_dir[1:])
plot_vectors(origin, vector, label="Tq", color="b")

### Fa tilde
vector = project_onto_plane(f, payload_dir[1:])
plot_vectors(origin, vector, label="Fa_tilde", color="g")

if THREE_D:
    ax.axis('equal')
    ax.legend()
else:
    plt.axis("equal")
    plt.legend()
plt.show()
