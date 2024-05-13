import cfusdlog
import matplotlib.pyplot as plt
from residual_calculation import residual


data_path = "./nn_logs/nn_log00"
data = cfusdlog.decode(data_path)['fixedFrequency']
print(data.keys())

timestamp = data["timestamp"][1:]
f, tau = residual(data)

fig, axs = plt.subplots(3, 1, figsize=(8, 6))
for i, v in enumerate(["x", "y", "z"]):
    axs[i].plot(timestamp, data[f"nn_output.f_{v}"][1:], label="residual prediction")
    axs[i].plot(timestamp, f[:, i], label=f"Force-{v} residual")
    axs[i].set_title(f"Force-{v} residual")
plt.tight_layout()
plt.legend()
plt.show()

fig, axs = plt.subplots(3, 1, figsize=(8, 6))
for i, v in enumerate(["x", "y"]): #, "z"]):
    axs[i].plot(timestamp, data[f"nn_output.tau_{v}"][1:], label="residual prediction")
    axs[i].plot(timestamp, tau[:, i])
    axs[i].set_title(f"Torque-{v} residual")
plt.tight_layout()
plt.show()
