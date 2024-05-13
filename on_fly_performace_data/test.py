import json
import matplotlib.pyplot as plt

model_c = json.load(open("./c_scaled.json"))
model_pth = json.load(open("./output_scaled.json"))[:100]
real_res = json.load(open("./real_res.json"))

for i in range(6):
    plt.plot([c[i] for c in model_c])
    plt.plot([c[i] for c in model_pth])
    plt.plot([c[i] for c in real_res])
    plt.show()
