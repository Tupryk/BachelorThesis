import cfusdlog
import numpy as np
import uav_trajectory
import matplotlib.pyplot as plt


traj = uav_trajectory.Trajectory()
traj.loadcsv("./figure8.csv")

traj.stretchtime(2)

data_paths = ['./data/controller_lee_only_force_no_scaling', './data/only_force_scaled', './data/standard_controller_2']

ts = np.arange(0, traj.duration, 0.01)
evals = np.empty((len(ts), 15))
for t, i in zip(ts, range(0, len(ts))):
    e = traj.eval(t)
    e.pos += np.array([0, 0, 1])
    evals[i, 0:3] = e.pos

plt.plot(evals[:, 0], evals[:, 1], label="Goal path")

for data_path in data_paths:
    data = cfusdlog.decode(data_path)['fixedFrequency']

    x = [i for i in data["stateEstimate.x"]]
    y = [i for i in data["stateEstimate.y"]]
    # z = [i for i in data["stateEstimate.z"]]
    label = data_path.replace("./data/", "").replace("_", " ")
    plt.plot(x, y, label=label)

plt.legend()
plt.axis('equal')
plt.show()
