import cfusdlog
import numpy as np
import uav_trajectory
import matplotlib.pyplot as plt


traj = uav_trajectory.Trajectory()
traj.loadcsv("./figure8.csv")

traj.stretchtime(2)

ts = np.arange(0, traj.duration, 0.01)
evals = np.empty((len(ts), 15))
for t, i in zip(ts, range(0, len(ts))):
    e = traj.eval(t)
    e.pos += np.array([0, 0, 1])
    evals[i, 0:3] = e.pos

data = cfusdlog.decode('./log06')['fixedFrequency']
x = [i for i in data["stateEstimate.x"]]
y = [i for i in data["stateEstimate.y"]]
z = [i for i in data["stateEstimate.z"]]
ax = plt.axes(projection="3d")
ax.plot(evals[:, 0], evals[:, 1], evals[:, 2])
ax.plot(x, y, z)
plt.axis('equal')
plt.show()
