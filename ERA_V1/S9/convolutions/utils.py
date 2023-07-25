import torch
import torch.nn as nn


class BaseDWSeparableConv2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        bias=False,
    ):
        super(BaseDWSeparableConv2d, self).__init__()

        # Depthwise convolution uses 3x3 kernel
        self.depthwise = nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels,  # Set groups to in_channels for depthwise convolution
            dilation=dilation,
            bias=False,  # Set bias to False for depthwise convolution
        )

        # Pointwise convolution uses 1x1 kernel
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=bias,  # Set bias only in pointwise convolution
        )

    def __call__(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class BaseConvLayer2d(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=0,
        dilation=1,
        depthwise_separable=False,
        enable_ReLU=False,
        norm="none",
        group_size=1,
        dropout=0.0,
        bias=False,
    ):
        super(BaseConvLayer2d, self).__init__()
        self.relu = enable_ReLU
        self.norm = norm
        self.dropout = dropout

        if depthwise_separable:
            self.convBlock2d = BaseDWSeparableConv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            )
        else:
            self.convBlock2d = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
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

    def forward(self, X):
        X = self.convBlock2d(X)
        if self.relu:
            x = self.relu(X)
        if self.norm:
            X = self.norm(X)
        if self.dropout:
            X = self.dropout(X)
        return X


class GenericConvLayer2d(BaseConvLayer2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=0,
        dilation=1,
        enable_ReLU=False,
        norm="none",
        group_size=1,
        dropout=0.0,
        bias=False,
    ):
        super(GenericConvLayer2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            depthwise_separable=False,
            enable_ReLU=enable_ReLU,
            norm=norm,
            group_size=group_size,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, X):
        return super(GenericConvLayer2d, self).forward(X)


class DWSeparableConv2d(BaseConvLayer2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        padding=0,
        dilation=1,
        enable_ReLU=False,
        norm="none",
        group_size=1,
        dropout=0.0,
        bias=False,
    ):
        super(DWSeparableConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            depthwise_separable=True,
            enable_ReLU=enable_ReLU,
            norm=norm,
            group_size=group_size,
            dropout=dropout,
            bias=bias,
        )

    def forward(self, X):
        return super(DWSeparableConv2d, self).forward(X)


class DilateConv2d(BaseConvLayer2d):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(3, 3),
        stride=1,
        dilation=2,
        padding=0,
    ):
        super(DilateConv2d, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            depthwise_separable=False,
            enable_ReLU=False,
            norm=None,
            group_size=1,
            dropout=0.0,
            bias=False,
        )

    def forward(self, X):
        return super(DilateConv2d, self).forward(X)
