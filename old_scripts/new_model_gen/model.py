import os
import torch
import numpy as np
from torch import nn
import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, input_size: int=12, hidden_size: int=16, output_size: int=6, learning_rate: float=.0035):
        super(MLP, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        self.test_lossess = []
        self.train_lossess = []

        self.layers = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )

    def forward(self, x):
        pred = self.layers(x)
        return pred

    def train_loop(self, dataloader, loss_fn, optimizer):
        self.train()

        train_loss = []
        for batch, (X, y) in enumerate(dataloader):
            pred = self.forward(X)
            loss = loss_fn(pred, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.item()
            train_loss.append(loss)

        avg_train = np.array(train_loss).sum()/len(dataloader)

        print(f'avg. train loss: {avg_train :> 5f}')
        return avg_train

    def test_loop(self, dataloader, loss_fn):
        num_batches = len(dataloader)
        test_loss = 0

        with torch.no_grad():
            for X, y in dataloader:
                pred = self.forward(X)
                test_loss += loss_fn(pred, y).item()

        test_loss /= num_batches

        print(f"Avg. test loss: {test_loss:>8f} \n")
        return test_loss 

    def train_model(self, train_dataloader, test_dataloader, epochs: int=32):

        self.double()
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        loss_fn = nn.MSELoss()
        self.train_losses = []
        self.test_losses = []

        for t in range(epochs):
            print(f"Epoch {t+1}\n-------------------------------")
            
            self.train_losses.append(self.train_loop(train_dataloader, loss_fn, optimizer))
            self.test_losses.append(self.test_loop(test_dataloader, loss_fn))
        
            print(f"\n-------------------------------")

        print("Done!")

    def show_progress(self):
        plt.plot(self.train_losses, label="Train lossess")
        plt.plot(self.test_losses, label="Test lossess")
        plt.legend()
        plt.show()
    
    def save(self, data_path: str="model.pth"):
        torch.save(self.state_dict(), f"./{data_path}")
