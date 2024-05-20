import cfusdlog
import matplotlib.pyplot as plt
from residual_calculation import residual


data_path = "./residual_comparison/nn_log00"
data = cfusdlog.decode(data_path)['fixedFrequency']
timestamp = data["timestamp"][1:]
f, tau = residual(data)

fig, axs = plt.subplots(3, 1, figsize=(8, 6))
for i, v in enumerate(["x", "y", "z"]):
    axs[i].plot(timestamp, data[f"stateEstimate.v{v}"][1:], label=f"Velocity in {v}")
    axs[i].plot(timestamp, f[:, i], label=f"Force residual in {v}")
    axs[i].legend()

plt.tight_layout()
plt.show()
