import rowan
import cfusdlog
import numpy as np
import matplotlib.pyplot as plt


data = cfusdlog.decode("./payload_data/cf3_t_03")['fixedFrequency']
print(data.keys())

accs = []
se_accs = []
for i in range(len(data["timestamp"])):
    quat = np.array([data['stateEstimate.qw'][i], data['stateEstimate.qx'][i],
                        data['stateEstimate.qy'][i], data['stateEstimate.qz'][i]])
    R = rowan.to_matrix(quat)

    acc = R @ np.array([data['acc.x'][i], data['acc.y'][i], data['acc.z'][i]])
    acc[2] -= 1.
    #acc *= 9.81
    accs.append(acc)
    se_accs.append(np.array([data['stateEstimate.ax'][i], data['stateEstimate.ay'][i], data['stateEstimate.az'][i]]))

accs = np.array(accs)
se_accs = np.array(se_accs)
for j in range(3):
    plt.plot(accs[:, j], label="calc_world", alpha=.5)
    plt.plot(se_accs[:, j], label="se_world", alpha=.5)
    plt.legend()
    plt.show()
