from torch import nn


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=12, hidden_size=15,  output_size=6):
        super(NeuralNetwork, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_relu = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, 10),
            nn.ReLU(),
            nn.Linear(10, self.output_size)
        )
    def forward(self, x):
        pred = self.linear_relu(x)
        return pred
    