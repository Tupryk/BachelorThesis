import numpy as np
import matplotlib.pyplot as plt

import uav_trajectory

traj = uav_trajectory.Trajectory()
traj.loadcsv("./figure8.csv")

traj.stretchtime(2)

ts = np.arange(0, traj.duration, 0.01)
evals = np.empty((len(ts), 15))
for t, i in zip(ts, range(0, len(ts))):
    e = traj.eval(t)
    evals[i, 0:3] = e.pos

ax = plt.axes(projection="3d")
ax.plot(evals[:, 0], evals[:, 1], evals[:, 2])

plt.show()
