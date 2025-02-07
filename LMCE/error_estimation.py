import numpy as np
from typing import List
import matplotlib.pyplot as plt


def error_calculator(real_pos: List[float], target_pos: List[float], cutoff_start: int=0, cutoff_end: int=-1, vis: bool = False) -> float:
    errors = []

    x = real_pos[0][cutoff_start:cutoff_end]
    y = real_pos[1][cutoff_start:cutoff_end]

    x_ = [target_pos[0][int(j/len(x)*len(target_pos[0]))]-x[j]
          for j, _, in enumerate(x)]
    y_ = [target_pos[1][int(j/len(x)*len(target_pos[1]))]-y[j]
          for j, _, in enumerate(x)]

    vector = np.array([x_, y_]).T

    acumulated_error = 0
    for j, _ in enumerate(x_):
        acumulated_error += np.linalg.norm(vector[j])
    errors.append(acumulated_error/len(x_))
    error = sum(errors)/len(errors)

    if vis:
        origin = np.array([x, y]).T
        vector = np.array([x_, y_]).T
        plt.plot(target_pos[0], target_pos[1], label="Desired path")
        plt.plot(x, y, label="Real path")
        plt.quiver(origin[:, 0], origin[:, 1], vector[:, 0], vector[:, 1], angles='xy',
                   scale_units='xy', scale=1, color='r', alpha=.1, label="Deviation from flight path")
        plt.xlabel("X position (m)")
        plt.ylabel("Y position (m)")
        plt.axis('equal')
        plt.legend()
        plt.savefig("error.png", dpi=600)
        plt.show()

    return error


def find_best_cutoff(real_pos: List[float], target_pos: List[float]) -> int:
    # Kind of slow, could be made nlogn
    smallest_error = np.inf
    start_cutoff = 0
    end_cutoff = -1
    current_error = 0

    path_len = len(real_pos[0])

    for _ in range(4):
        smallest_error = np.inf
        current_error = 0
        for cutoff in range(0, path_len - path_len//3):
            current_error = error_calculator(real_pos, target_pos, cutoff, end_cutoff)
            if smallest_error > current_error:
                smallest_error = current_error
                start_cutoff = cutoff

        for cutoff in range(len(real_pos[0]), path_len//3, -1):
            current_error = error_calculator(real_pos, target_pos, start_cutoff, cutoff)
            if smallest_error > current_error:
                smallest_error = current_error
                end_cutoff = cutoff

    return start_cutoff, end_cutoff


if __name__ == "__main__":
    import cfusdlog
    import uav_trajectory

    # Get desired path
    traj = uav_trajectory.Trajectory()
    traj.loadcsv("./flight_paths/figure8.csv")
    traj.stretchtime(2)

    ts = np.arange(0, traj.duration, 0.01)
    evals = np.empty((len(ts), 15))
    for t, i in zip(ts, range(0, len(ts))):
        e = traj.eval(t)
        e.pos += np.array([0, 0, 1])
        evals[i, 0:3] = e.pos

    target_pos = evals.transpose()

    # Get real path
    data_path = "../crazyflie-data-collection/brushless_flights/data/eckart02"
    data = cfusdlog.decode(data_path)['fixedFrequency']
    real_pos = [data["stateEstimate.x"], data["stateEstimate.y"]]

    # Calculate error
    errors = []
    for i in range(1, len(real_pos[0])):
        cutoff = i
        error = error_calculator(cutoff, real_pos, target_pos)
        errors.append(error)

    plt.plot(errors)
    plt.show()

    print(find_best_cutoff(real_pos, target_pos))
