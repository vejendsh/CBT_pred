import torch.nn as nn
import torch

class NeuralNetwork(nn.Module):
    def __init__(self, out_features: int = 102):
        super().__init__()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(5, 32),
            nn.Tanh(),
            nn.Linear(32, 64),
            nn.Tanh(),
            nn.Linear(64, 128),
            nn.Tanh(),
            nn.Linear(128, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 32),
            nn.Tanh(),
            nn.Linear(32, out_features),
            nn.Tanh()
        )

        # for layer in self.linear_relu_stack:
        #     if isinstance(layer, nn.Linear):
        #         nn.init.xavier_uniform_(layer.weight, gain=nn.init.calculate_gain('tanh'))  # For uniform distribution
        #         nn.init.zeros_(layer.bias)  # Optional, bias initialized to zero


    def forward(self, x):
        pred = self.linear_relu_stack(x)
        return pred 
