import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot

import matplotlib.pyplot as plt
from tqdm import tqdm


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        padding=0,
        enable_ReLU=False,
        norm="none",
        group_size=2,
        dropout=0.0,
    ):
        super(ConvBlock, self).__init__()
        self.relu = enable_ReLU
        self.norm = norm
        self.dropout = dropout

        self.convBlock2d = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            bias=False,
        )

        if self.relu:
            self.relu = nn.ReLU()

        if self.norm == "bn":
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == "gn":
            self.norm = nn.GroupNorm(group_size, out_channels)
        elif norm == "ln":
            self.norm = nn.GroupNorm(1, out_channels)
        else:
            self.norm = None

        if self.dropout > 0:
            self.dropout = nn.Dropout(dropout)

    def __call__(self, X):
        X = self.convBlock2d(X)
        if self.relu:
            x = self.relu(X)
        if self.norm:
            X = self.norm(X)
        if self.dropout:
            X = self.dropout(X)
        return X


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self, norm="none", dropout=0.0, skip_connets=False):
        super(Net, self).__init__()
        self.norm = norm
        self.dropout = dropout
        self.skip_connets = skip_connets

        # Block 1
        self.convblock1 = ConvBlock(
            in_channels=3,
            out_channels=4,
            kernel_size=(3, 3),
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 4 x 32 x 32
        self.convblock2 = ConvBlock(
            in_channels=4,
            out_channels=8,
            kernel_size=(3, 3),
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 8 x 32 x 32

        # Transition Block1
        self.convblock3 = ConvBlock(
            in_channels=8,
            out_channels=8,
            kernel_size=(1, 1),
            padding=0,
            enable_ReLU=False,
        )  # 8 x 32 x 32
        self.pool1 = nn.MaxPool2d(2, 2)  # 8 x 16 x 16

        # Block 2
        self.convblock4 = ConvBlock(
            in_channels=8,
            out_channels=16,
            kernel_size=(3, 3),
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 16 x 16 x 16
        self.convblock5 = ConvBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 32 x 16 x 16

        # Transition Block2
        self.convblock6 = ConvBlock(
            in_channels=32,
            out_channels=16,
            kernel_size=(1, 1),
            padding=0,
            enable_ReLU=False,
        )  # 16 x 16 x 16
        self.pool2 = nn.MaxPool2d(2, 2)  # 16 x 8 x 8

        # Block 3
        self.convblock7 = ConvBlock(
            in_channels=16,
            out_channels=32,
            kernel_size=(3, 3),
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 32 x 8 x 8

        self.convblock8 = ConvBlock(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 64 x 8 x 8

        # Transition Block3
        self.convblock9 = ConvBlock(
            in_channels=64,
            out_channels=32,
            kernel_size=(1, 1),
            padding=0,
            enable_ReLU=False,
        )  # 32 x 8 x 8
        # GAP
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=8))  # 32 x 1 x 1

        # Output Block
        self.convblock10 = ConvBlock(
            in_channels=32,
            out_channels=10,
            kernel_size=(1, 1),
            padding=0,
            enable_ReLU=False,
        )  # 10 x 1 X 1

    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)
        x = self.convblock3(x)
        x = self.pool1(x)
        x = self.convblock4(x)
        x = self.convblock5(x)
        x = self.convblock6(x)
        x = self.pool2(x)
        x = self.convblock7(x)
        x = self.convblock8(x)
        x = self.convblock9(x)
        x = self.gap(x)
        x = self.convblock10(x)

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1)

    def model_summary(self, input_size):
        device = next(self.parameters()).device
        print(
            f"Device: {device}\n"
            f"Normalization: {self.norm}\n"
            f"Dropout: {self.dropout}\n"
            f"Skip Connection: {self.skip_connets}"
        )
        summary(self, input_size=input_size)

    def model_visualize(self, device, loader, png_name, format="png"):
        batch = next(iter(loader))
        self.eval()
        yhat = self(batch[0].to(device))  # Give dummy batch to forward().

        make_dot(yhat, params=dict(list(self.named_parameters()))).render(
            png_name, format=format
        )
