import torch
import cfusdlog
import numpy as np
import matplotlib.pyplot as plt
from model import NeuralNetwork
from residual_calculation import residual
from data_to_model import convert, scale_outputs
import json


data_path = "./nn_logs/nn_log01"
data = cfusdlog.decode(data_path)['fixedFrequency']
model_input = convert(data)
json.dump(model_input.numpy().tolist()[:100], open("./input.json", "w"))

timestamp = data["timestamp"][1:]
f, tau = residual(data)
real = [[*f_, *tau[i]] for i, f_ in enumerate(f)]
json.dump(real[:100], open("./real_res.json", "w"))

original_model_output = []
model = NeuralNetwork()
model.load_state_dict(torch.load("../pth_models/jana_nn.pth"))
for mi in model_input:
    original_model_output.append(scale_outputs(model.forward(mi).detach().numpy()).tolist())
json.dump(original_model_output[:100], open("./output_scaled.json", "w"))

fig, axs = plt.subplots(3, 1, figsize=(8, 6))
for i, v in enumerate(["x", "y", "z"]):
    axs[i].plot(timestamp, [original_model_output[j][i] for j in range(len(original_model_output))], label="residual prediction offline")
    axs[i].plot(timestamp, data[f"nn_output.f_{v}"][1:], label="residual prediction on-flight")
    axs[i].plot(timestamp, f[:, i], label=f"Force real residual")
axs[i].set_title(f"Force residual")
plt.legend()
plt.tight_layout()
plt.show()

fig, axs = plt.subplots(2, 1, figsize=(8, 6))
for i, v in enumerate(["x", "y"]): #, "z"]):
    axs[i].plot(timestamp, [original_model_output[j][i+3] for j in range(len(original_model_output))], label="residual prediction offline")
    axs[i].plot(timestamp, data[f"nn_output.tau_{v}"][1:], label="residual prediction offline")
    axs[i].plot(timestamp, tau[:, i], label=f"Torque real residual")
axs[i].set_title(f"Torque real residual")
plt.legend()
plt.tight_layout()
plt.show()
