import cfusdlog
import matplotlib.pyplot as plt
from data_to_model import convert


data_path = "./input_logs/se/nn_log03"
data = cfusdlog.decode(data_path)['fixedFrequency']
timestamp = data["timestamp"][1:]
real_model_input = convert(data)
logged_model_input = data

params = ["se_acc_x", "se_acc_y", "se_acc_z", "gyro_x", "gyro_y", "gyro_z", "se_r_0", "se_r_1", "se_r_2", "se_r_3", "se_r_4", "se_r_5"]
for i, param in enumerate(params):
    try:
        plt.plot(timestamp, logged_model_input[f"nn_input.{param}"][1:], label="logged")
        plt.plot(timestamp, [real_model_input[j][i] for j, _ in enumerate(real_model_input)], label="real")
        plt.title(param)
        plt.legend()
        plt.show()
    except: pass
