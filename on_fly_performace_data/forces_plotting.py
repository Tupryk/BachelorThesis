import cfusdlog
import numpy as np
import uav_trajectory
import matplotlib.pyplot as plt
from residual_calculation import residual, residual_v2, brushless_residual


THREE_D = False

if THREE_D:
    fig = plt.figure()
    ax = plt.axes(projection ='3d')

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
data_paths = ["../crazyflie-data-collection/brushless_flights/data/eckart34"]
labels = ["eckart34"]

for i, data_path in enumerate(data_paths):
    data = cfusdlog.decode(data_path)['fixedFrequency']
    print(data.keys())

    x = [i for i in data["stateEstimate.x"]]
    y = [i for i in data["stateEstimate.y"]]
    z = [i-1. for i in data["stateEstimate.z"]]

    f, _ = brushless_residual(data, use_rpm=True)
    origin = np.array([x, y]).T
    vector = np.array([f[:, 0], f[:, 1]]).T * 10
    plt.quiver(origin[:,0], origin[:,1], vector[:,0], vector[:,1], angles='xy', scale_units='xy', scale=1, color='g', alpha=.1, label="Residual with rpm scaled by 10")
    f, _ = brushless_residual(data, use_rpm=False)
    origin = np.array([x, y]).T
    vector = np.array([f[:, 0], f[:, 1]]).T * 10
    plt.quiver(origin[:,0], origin[:,1], vector[:,0], vector[:,1], angles='xy', scale_units='xy', scale=1, color='r', alpha=.1, label="Residual with pwm scaled by 10")

    if THREE_D:
        ax.plot3D(x, y, z, label=labels[i])
    else:
        plt.plot(x, y, label=labels[i])

    f, tau = residual(data)
    for j, v in enumerate(["x", "y", "z"]):
        f_j = [f_[j] for f_ in f]
        print(f"{labels[i]} mean residual f_{v}: {sum(f_j)/len(f_j)}")

plt.title("Fx and Fy compared to quadrotor trajectory")
plt.legend()
plt.axis('equal')
plt.show()
