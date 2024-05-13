import torch
from model import NeuralNetwork

model = NeuralNetwork()
model.load_state_dict(torch.load("../pth_models/jana_nn.pth"))
output = model.forward(torch.tensor([ .1, -1., 3.5, -.05, -2., .04, .14, 0.1, 2.5, -.1, .4, 0.9 ]))
print(output)
