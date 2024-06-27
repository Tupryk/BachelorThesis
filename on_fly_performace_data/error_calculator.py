import cfusdlog
import numpy as np
import uav_trajectory
import matplotlib.pyplot as plt
from residual_calculation import residual


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

target_xs = evals[:, 0]
target_ys = evals[:, 1]
plt.plot(target_xs, target_ys, label="Desired path")
# plt.plot(target_xs, label="Desired path")

### Recorded data ###
# data_paths = [f"./error_calculating/nn/nn_log0{i}" for i in range(10)]
# data_paths = [f"./error_calculating/lee/nn_log0{i}" for i in range(1, 9)]
data_paths = [f"./error_calculating/lee/nn_log0{i}" for i in range(1)]
labels = data_paths

cutoff = 2700
errors = []
for i, data_path in enumerate(data_paths):
    data = cfusdlog.decode(data_path)['fixedFrequency']

    x = [i for i in data["stateEstimate.x"]][:cutoff]
    y = [i for i in data["stateEstimate.y"]][:cutoff]
    z = [i-1. for i in data["stateEstimate.z"]][:cutoff]

    origin = np.array([x, y]).T
    x_ = [target_xs[int(j/len(x)*len(target_xs))]-x[j] for j, _, in enumerate(x)]
    y_ = [target_ys[int(j/len(x)*len(target_xs))]-y[j] for j, _, in enumerate(x)]
    z_ = [-z[j] for j, _, in enumerate(x)]

    # target_xs_adjusted = [target_xs[int(j/len(x)*len(target_xs))] for j, _, in enumerate(x)]
    # plt.plot(target_xs_adjusted, label="Desired path")

    vector = np.array([x_, y_]).T
    plt.quiver(origin[:,0], origin[:,1], vector[:,0], vector[:,1], angles='xy', scale_units='xy', scale=1, color='r', alpha=.1)

    acumulated_error = 0
    for j, _ in enumerate(x_):
        print(np.linalg.norm(vector))
        acumulated_error += np.linalg.norm(vector[j])
    errors.append(acumulated_error/len(x_))
    print("Acumulated error: ", acumulated_error)
    print("Average error: ", acumulated_error/len(x_))

    if THREE_D:
        ax.plot3D(x, y, z, label=labels[i])
    else:
        plt.plot(x, y, label=labels[i])

    f, tau = residual(data)
    for j, v in enumerate(["x", "y", "z"]):
        f_j = [f_[j] for f_ in f]
        mean = sum(f_j)/len(f_j)
        print(f"{labels[i]} mean residual f_{v}: {mean}")

print(errors)
mean = sum(errors)/len(errors)
stds = [e - mean for e in errors]
std = sum(stds)/len(stds)
print("Mean error (m): ", mean)
print("Std. deviation (m): ", std)

plt.title("Fx and Fy compared to quadrotor trajectory")
plt.legend()
plt.axis('equal')
plt.show()
