import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot

from convolutions.utils import GenericConvLayer2d
from convolutions.utils import DWSeparableConv2d
from convolutions.utils import DilateConv2d


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self, norm="none", dropout=0.0, skip_connets=False):
        super(Net, self).__init__()
        self.norm = norm
        self.dropout = dropout
        # self.skip_connets = skip_connets
        self.skip_connets = False

        # Block 1
        self.convblock1 = DWSeparableConv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 32 x 32 x 32 | RF 3 | J_in = 1 & J_out = 1
        self.convblock2 = DWSeparableConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 64 x 32 x 32 | RF 5 | J_in = 1 & J_out = 1
        self.convblock3 = DilateConv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            stride=2,
            padding=0,
            dilation=2,
        )  # 32 x 14 x 14 | RF 11 | J_in = 1 & J_out = 1

        # Block 2
        self.convblock4 = DWSeparableConv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 64 x 14 x 14 | RF 13 | J_in = 1 & J_out = 1
        self.convblock5 = DWSeparableConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 128 x 14 x 14 | RF 15 | J_in = 1 & J_out = 1
        self.convblock6 = GenericConvLayer2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            stride=2,
            padding=0,
            enable_ReLU=False,
        )  # 64 x 6 x 6 | RF 17 | J_in = 1 & J_out = 2

        # Block 3
        self.convblock7 = DWSeparableConv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 128 x 6 x 6 | RF 21 | J_in = 2 & J_out = 2
        self.convblock8 = DWSeparableConv2d(
            in_channels=128,
            out_channels=128,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 128 x 6 x 6 | RF 25 | J_in = 2 & J_out = 2
        self.convblock9 = GenericConvLayer2d(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            stride=2,
            padding=0,
            enable_ReLU=False,
        )  # 64 x 2 x 2 | RF 29 | J_in = 2 & J_out = 4

        # Block 4
        self.convblock10 = DWSeparableConv2d(
            in_channels=64,
            out_channels=64,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 64 x 2 x 2 | RF 37 | J_in = 4 & J_out = 4
        self.convblock11 = DWSeparableConv2d(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            stride=1,
            padding=1,
            enable_ReLU=True,
            norm=self.norm,
            dropout=self.dropout,
        )  # 32 x 2 x 2 | RF 45 | J_in = 4 & J_out = 4
        self.convblock12 = GenericConvLayer2d(
            in_channels=32,
            out_channels=10,
            kernel_size=(1, 1),
            stride=1,
            padding=0,
            enable_ReLU=False,
        )  # 10 x 2 x 2 | RF 45 | J_in = 4 & J_out = 4

        # GAP
        self.gap = nn.Sequential(nn.AvgPool2d(kernel_size=2))  # 10 x 1 x 1

    def forward(self, x):  # X = 3 x 32 x 32
        # Block 1
        x = self.convblock1(x)  # 32 x 32 x 32 | RF 3 | J_in = 1 & J_out = 1
        x = self.convblock2(x)  # 64 x 32 x 32 | RF 5 | J_in = 1 & J_out = 1
        x = self.convblock3(x)  # 32 x 14 x 14 | RF 11 | J_in = 1 & J_out = 1

        # Block 2
        x = self.convblock4(x)  # 64 x 14 x 14 | RF 13 | J_in = 1 & J_out = 1
        x = self.convblock5(x)  # 128 x 14 x 14 | RF 15 | J_in = 1 & J_out = 1
        x = self.convblock6(x)  # 64 x 6 x 6 | RF 17 | J_in = 1 & J_out = 2

        # Block 3
        x = self.convblock7(x)  # 128 x 6 x 6 | RF 21 | J_in = 2 & J_out = 2
        x = self.convblock8(x)  # 128 x 6 x 6 | RF 25 | J_in = 2 & J_out = 2
        x = self.convblock9(x)  # 64 x 2 x 2 | RF 29 | J_in = 2 & J_out = 4

        # Block 4
        x = self.convblock10(x)  # 64 x 2 x 2 | RF 37 | J_in = 4 & J_out = 4
        x = self.convblock11(x)  # 32 x 2 x 2 | RF 45 | J_in = 4 & J_out = 4
        x = self.convblock12(x)  # 10 x 2 x 2 | RF 45 | J_in = 4 & J_out = 4

        # GAP
        x = self.gap(x)  # 10 x 1 x 1

        x = x.view(-1, 10)
        return F.log_softmax(x, dim=-1) # 10 x 1

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
