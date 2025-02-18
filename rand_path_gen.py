import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from noise import pnoise1
import matplotlib.colors as mcolors
import matplotlib.cm as cm


def gen_rand_flight_path(bbox: list[list[float]], scale: float=.5, resolution: int=100, seed: int=None):

    if seed == None:
        seed = np.random.randint(0, 100_000)
    print("Seed: ", seed)
    np.random.seed(seed)

    ### Generate path ###
    t = np.linspace(0, 10, resolution)
    linear_term = np.random.uniform(0.05, 0.5, 3)
    linear_sign = np.random.choice([-1, 1], 3)
    x = [pnoise1(i * scale, repeat=1024) + i*linear_term[0]*linear_sign[0] for i in t]
    y = [pnoise1(i * scale + 250, repeat=1024) + i*linear_term[1]*linear_sign[1] for i in t]
    z = [pnoise1(i * scale + 500, repeat=1024) + i*linear_term[2]*linear_sign[2] for i in t]

    traj = np.array([x, y, z]).T

    ### Fit into bounding box ###
    center = np.array([
        (max(x)+min(x))*.5,
        (max(y)+min(y))*.5,
        (max(z)+min(z))*.5
    ])
    traj -= center

    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]

    min_x = min(x)
    min_y = min(y)
    min_z = min(z)
    max_x = max(x)
    max_y = max(y)
    max_z = max(z)

    traj[:, 0] = (x - min_x) / (max_x - min_x) * (bbox[1][0] - bbox[0][0]) + bbox[0][0]
    traj[:, 1] = (y - min_y) / (max_y - min_y) * (bbox[1][1] - bbox[0][1]) + bbox[0][1]
    traj[:, 2] = (z - min_z) / (max_z - min_z) * (bbox[1][2] - bbox[0][2]) + bbox[0][2]

    return traj


if __name__ == "__main__":

    bbox = [[-1.0, -1.0,  0.1],
            [ 1.0,  1.0,  1.0]]

    traj = gen_rand_flight_path(bbox, resolution=50)
    x = traj[:, 0]
    y = traj[:, 1]
    z = traj[:, 2]

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create a gradient colormap
    norm = plt.Normalize(0, len(x))
    cmap = cm.get_cmap('coolwarm')
    colors = [cmap(norm(i)) for i in range(len(x))]
    
    for i in range(len(x) - 1):
        ax.plot(x[i:i+2], y[i:i+2], z[i:i+2], color=colors[i], lw=2)
    
    # Create a colorbar legend
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.5, aspect=10)
    cbar.set_label('Path Progression (Blue: Start, Red: End)')
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Natural 3D Path with Gradient')
    # plt.show()

    scale = np.random.random()
    yaw = np.sin(np.linspace(0, np.pi*2*scale, len(x))) * np.pi*2
    # plt.plot(yaw)
    # plt.show()

    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'uav_trajectories/scripts')))
    from generate_trajectory import generate_trajectory

    pieces = 5
    with open("in.csv".format(i), "w") as f:
        f.write("t,x,y,z,yaw\n")
        ts = np.linspace(0, 10, len(x))
        for i, t in enumerate(ts):
            f.write("{},{},{},{},{}\n".format(t, x[i], y[i], z[i], yaw[i]))
    data = np.loadtxt("in.csv", delimiter=',', skiprows=1)
    traj = generate_trajectory(data, pieces)
    traj.savecsv("out.csv")

    import LMCE.uav_trajectory as uav_trajectory

    traj = uav_trajectory.Trajectory()
    traj.loadcsv("./out.csv")

    ts = np.arange(0, traj.duration, 0.01)
    evals = np.empty((len(ts), 15))
    for t, i in zip(ts, range(0, len(ts))):
        e = traj.eval(t)
        e.pos += np.array([0, 0, 1])
        evals[i, 0:3] = e.pos

    target_pos = evals.transpose()[:3]
    target_pos[2] -= 1.
    ax.plot(*target_pos, color="green")

    plt.show()
