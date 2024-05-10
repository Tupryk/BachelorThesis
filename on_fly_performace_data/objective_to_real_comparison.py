import cfusdlog
import numpy as np
import uav_trajectory
import matplotlib.pyplot as plt


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
data_paths = ["./slow/lee/log00", "./slow/nnscaledwithtorques/log00"]
labels = ["Standard Lee controller", "Lee ctrl. + NN"]

for i, data_path in enumerate(data_paths):
    data = cfusdlog.decode(data_path)['fixedFrequency']

    x = [i for i in data["stateEstimate.x"]]
    y = [i for i in data["stateEstimate.y"]]
    z = [i-1. for i in data["stateEstimate.z"]]
    print(len(x))
    # label = data_path.replace("./data/", "").replace("_", " ")
    if THREE_D:
        ax.plot3D(x, y, z, label=labels[i])
    else:
        plt.plot(x, y, label=labels[i])

plt.legend()
plt.axis('equal')
plt.show()
