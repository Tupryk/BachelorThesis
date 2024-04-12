import numpy as np
import torch
from torch import nn
import matplotlib.pyplot as plt


class NeuralNetwork(nn.Module):
    def __init__(self, input_size=12, hidden_size=15,  output_size=6):
        super(NeuralNetwork, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.linear_relu = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        pred = self.linear_relu(x)
        return pred

    def train_loop(self, dataloader, loss_fn, optimizer):
        self.train()
        d = 0
        train_loss = []
        for batch, (X, y) in enumerate(dataloader):
            pred = self.forward(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch % 100 == 0:
                d += 1
                loss = loss.item()
                train_loss.append(loss)

        avg_train = np.array(train_loss).sum()/d

        print(f'Avg. train loss: {avg_train :> 7f}')
        return avg_train

    def test_loop(self, dataloader, loss_fn):
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self.forward(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches

        print(f"Avg. test loss: {test_loss :> 7f} \n")
        return test_loss

    def train_model(self, train_dataloader, test_dataloader):

        self.double()
        epoch = 30
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0035)

        loss_fn = nn.MSELoss()
        train_losses = []
        test_losses = []

        for t in range(epoch):
            print(f"Epoch {t+1}\n-------------------------------")

            train_losses.append(self.train_loop(
                train_dataloader, loss_fn, optimizer))
            test_losses.append(self.test_loop(test_dataloader, loss_fn))

            print(f"\n-------------------------------")

        print("Done!")
        train_losses = np.array(train_losses)
        test_losses = np.array(test_losses)

        plt.plot(train_losses)
        plt.plot(test_losses)
        plt.show()

        # torch.save(self.state_dict(), 'model_1.pth')
