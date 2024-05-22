import cfusdlog
import numpy as np
import matplotlib.pyplot as plt
from residual_calculation import residual


data_path = "./new_data/nn_log12"
data = cfusdlog.decode(data_path)['fixedFrequency']

f, _ = residual(data)
forces = [np.linalg.norm(f_) for f_ in f]
plt.plot(data["timestamp"][1:], forces)

plt.title("Forces over time in linear path")
plt.xlabel("Timestamp")
plt.ylabel("Force")
plt.show()
