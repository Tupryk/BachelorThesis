import numpy as np
import matplotlib.pyplot as plt

import LMCE.cfusdlog as cfusdlog
import LMCE.uav_trajectory as uav_trajectory
from LMCE.residual_calculation import residual
from LMCE.error_estimation import error_calculator, find_best_cutoff


traj = uav_trajectory.Trajectory()
traj.loadcsv("./LMCE/flight_paths/figure8.csv")

traj.stretchtime(2)

ts = np.arange(0, traj.duration, 0.01)
evals = np.empty((len(ts), 15))
for t, i in zip(ts, range(0, len(ts))):
    e = traj.eval(t)
    e.pos += np.array([0, 0, 1])
    evals[i, 0:3] = e.pos

plt.plot(evals[:, 0], evals[:, 1], label="Desired path")

data_path = "./timescale7/cf21/standard/nn_log10"

data = cfusdlog.decode(data_path)['fixedFrequency']

origin = np.array([data["stateEstimate.x"], data["stateEstimate.y"]]).T

scaling = 10

f, _ = residual(data, use_rpm=False)

vector = np.array([f[:, 0], f[:, 1]]).T * scaling
plt.quiver(origin[:, 0], origin[:, 1], vector[:, 0], vector[:, 1], angles='xy',
            scale_units='xy', scale=1, color='r', alpha=.1, label=f"Post-flight calculations (scaled by {scaling})")

vector = np.array([data["nn_output.f_x"], data["nn_output.f_y"]]).T * scaling
plt.quiver(origin[:, 0], origin[:, 1], vector[:, 0], vector[:, 1], angles='xy',
            scale_units='xy', scale=1, color='g', alpha=.1, label=f"Prediction (scaled by {scaling})")

plt.plot(data["stateEstimate.x"], data["stateEstimate.y"], label="Real path")

plt.title("Fx and Fy compared to quadrotor trajectory")
plt.legend()
plt.axis('equal')
plt.show()

plt.plot(f[:, 0], label="Post-flight calculations")
plt.plot(data["nn_output.f_x"], label="In flight predictions")
plt.title("Post-flight Fx compared to predictions in flight")
plt.legend()
plt.show()

plt.plot(f[:, 1], label="Post-flight calculations")
plt.plot(data["nn_output.f_y"], label="In flight predictions")
plt.title("Post-flight Fy compared to predictions in flight")
plt.legend()
plt.show()

target_pos = evals.transpose()
real_pos = [data["stateEstimate.x"], data["stateEstimate.y"]]
cutoff = find_best_cutoff(real_pos, target_pos)
avg_error = error_calculator(cutoff, real_pos, target_pos, vis=True)
print(f"Avg. error: {avg_error:.4f} (m)")
