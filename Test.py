import torch
from autoencoder_weiver import full_network
from sindy_utils import library_size
import numpy as np
from examples.lorenz.example_lorenz import get_lorenz_data
import os
from autoencoder_weiver import define_loss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from Deep_Delay_AutoEncoder_Wei import SVD
from Deep_Delay_AutoEncoder_Wei import inverse_SVD
from Deep_Delay_AutoEncoder_Wei import linear_autoencoder

x = torch.Tensor(np.random.rand(10000,128))
dx = torch.Tensor(np.random.rand(10000,128))
x.requires_grad = True

v, U_p = SVD(x, 32)

import torch
# Define the encoder layer that computes Z from A
class Encoder(torch.nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.linear = torch.nn.Linear(10, 5)
        self.activation = torch.nn.ReLU()

    def forward(self, A):
        Z = self.linear(A)
        Z = self.activation(Z)
        return Z


# Create an instance of the encoder layer
encoder = Encoder()

# Create the input tensor A and enable gradient tracking
A = torch.randn(2, 10, requires_grad=True)

# Compute the output tensor Z
Z = encoder(A)

# Compute the gradient of Z with respect to A using the chain rule
dZ_dA = torch.autograd.grad(Z, A, torch.ones_like(Z), retain_graph=True)[0]

# Print the result
print(dZ_dA.shape)

