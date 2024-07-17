# Dnmodel.py
from thop import profile
import torch
import torch.nn as nn


# Define DnCNN model structure
class DnCNN(nn.Module):
    def __init__(self, channels=3, num_of_layers=17):
        super(DnCNN, self).__init__()
        kernel_size = 3
        padding = 1
        features = 64
        layers = []
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features, kernel_size=kernel_size, padding=padding,
                                bias=False))
        layers.append(nn.ReLU(inplace=True))
        for _ in range(num_of_layers - 2):
            layers.append(
                nn.Conv2d(in_channels=features, out_channels=features, kernel_size=kernel_size, padding=padding,
                          bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels, kernel_size=kernel_size, padding=padding,
                                bias=False))
        self.dncnn = nn.Sequential(*layers)

    def forward(self, x):
        out = self.dncnn(x)
        return out


if __name__ == "__main__":
    # Define the input tensor
    input = torch.ones(1, 3, 256, 256, dtype=torch.float, requires_grad=False)

    # Initialize the DnCNN model
    model = DnCNN()

    # Forward pass
    out = model(input)

    # Compute FLOPs and parameters
    flops, params = profile(model, inputs=(input,))

    # Print input shape, parameters, FLOPs, and output shape
    print('input shape:', input.shape)
    print('parameters:', params)
    print('flops:', flops)
    print('output shape:', out.shape)
