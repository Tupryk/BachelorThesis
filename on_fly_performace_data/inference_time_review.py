import cfusdlog
import numpy as np

data_path = "./inference_time_data/nn_log02"
data = cfusdlog.decode(data_path)['fixedFrequency']
print(data['nn_perf.inf_tim'])
print(f"Decision tree ensemble avg. inf. time: {np.mean(data['nn_perf.inf_tim'])} Hz")

data_path = "./inference_time_data/nn_log07"
data = cfusdlog.decode(data_path)['fixedFrequency']
print(data['nn_perf.inf_tim'])
print(f"Neural network avg. inf. time: {np.mean(data['nn_perf.inf_tim'])} Hz")
