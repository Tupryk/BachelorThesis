import cfusdlog
import numpy as np
import uav_trajectory
import matplotlib.pyplot as plt
from residual_calculation import residual, residual_v2


### Trajectory to be followed ###
traj = uav_trajectory.Trajectory()
traj.loadcsv("./figure8.csv")

traj.stretchtime(2)

ts = np.arange(0, traj.duration, 0.01)
evals = np.empty((len(ts), 15))
for t, i in zip(ts, range(0, len(ts))):
    e = traj.eval(t)
    e.pos += np.array([0, 0, 1])
    evals[i, 0:3] = e.pos

plt.plot(evals[:, 0], evals[:, 1], label="Desired path")

### Recorded data ###
# data_paths = ["./new_data/nn_log04"]
# labels = ["Lee ctrl. + NN"]
data_path = "../flight_data/jana02"
label = "jana02"

vel2res = []
projections = []
data = cfusdlog.decode(data_path)['fixedFrequency']
print(data.keys())

x = [i for i in data["stateEstimate.x"]]
y = [i for i in data["stateEstimate.y"]]
z = [i-1. for i in data["stateEstimate.z"]]

origin = np.array([x[1:], y[1:]]).T
vector_v = np.array([data["stateEstimate.vx"][1:], data["stateEstimate.vy"][1:]]).T
scale = .3
plt.quiver(origin[:,0], origin[:,1], vector_v[:,0] * scale, vector_v[:,1] * scale, angles='xy', scale_units='xy', scale=1, color='b', alpha=.1, label="velocities")

f, _ = residual(data, use_rpm=False, rot=True)
origin = np.array([x[1:], y[1:]]).T
vector_r = np.array([f[:, 0], f[:, 1]]).T * 5
plt.quiver(origin[:,0], origin[:,1], vector_r[:,0], vector_r[:,1], angles='xy', scale_units='xy', scale=1, color='r', alpha=.1, label="residuals")

plt.plot(x, y, label=label)

for j in range(len(vector_v)):
    vel = vector_v[j]/np.linalg.norm(vector_v[j])
    res = vector_r[j]/np.linalg.norm(vector_r[j])
    dot_product = np.dot(vel, res)

    projected = (dot_product / np.dot(vector_v[j], vector_v[j])) * vector_v[j]

    projections.append(projected)
    vel2res.append(dot_product)

projections = np.array(projections)

plt.title("Residuals (scaled by 5) and velocities (scaled by .3)")
plt.legend()
plt.axis('equal')
plt.show()

plt.plot(vel2res)
plt.title("Cosine of the angle between the velocity and residual in 2d space over time")
plt.show()

f = residual_v2(data)
origin = np.array([x[1:], y[1:]]).T
vector_r = np.array([f[:, 0], f[:, 1]]).T
plt.quiver(origin[:,0], origin[:,1], vector_r[:,0], vector_r[:,1], angles='xy', scale_units='xy', scale=1, color='b', alpha=.1, label="Drag model")
scale = .1
plt.quiver(origin[:,0], origin[:,1], projections[:,0] * scale, projections[:,1] * scale, angles='xy', scale_units='xy', scale=1, color='r', alpha=.1, label=f"Projected scaled by {scale}")
plt.title("Comparison between the projection of the residual on the velocity and the Foster drag model")
plt.legend()
plt.show()

plt.plot(vector_r[:,0], label="Drag model")
plt.plot(projections[:,0]*.01, label=f"Projected scaled by {.01}")
plt.ylim(min(vector_r[:,0]), max(vector_r[:,0]))
plt.title("projection of the residual on the velocity and drag model (x axis)")
plt.show()

plt.plot(vector_r[:,1], label="Drag model")
plt.plot(projections[:,1]*.01, label=f"Projected scaled by {.01}")
plt.ylim(min(vector_r[:,1]), max(vector_r[:,1]))
plt.title("projection of the residual on the velocity and drag model (y axis)")
plt.show()
