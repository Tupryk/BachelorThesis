import time
import subprocess

nn_times = []
tree_times = []
run_count = 100
for i in range(run_count):
    start_time = time.time()

    process = subprocess.Popen('../pth_to_c_converter/p', shell=True)
    process.wait()

    end_time = time.time()

    nn_inf_time = end_time - start_time
    nn_times.append(nn_inf_time)

    start_time = time.time()

    process = subprocess.Popen('../tree_to_c_converter/p', shell=True)
    process.wait()

    end_time = time.time()

    tree_inf_time = end_time - start_time
    tree_times.append(tree_inf_time)

print(f"--- Inference time averaged over {run_count} runs ---")
print(f"- Neural network: {sum(nn_times)/len(nn_times):.7f} seconds")
print(f"- Decision forest: {sum(tree_times)/len(tree_times):.7f} seconds")
