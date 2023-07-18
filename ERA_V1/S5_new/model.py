import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary
from torchviz import make_dot

import matplotlib.pyplot as plt
from tqdm import tqdm


class Net(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=(3, 3), padding=0, bias=False
        )
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), padding=0, bias=False
        )
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), padding=0, bias=False
        )
        self.conv4 = nn.Conv2d(
            in_channels=128, out_channels=256, kernel_size=(3, 3), padding=0, bias=False
        )
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def model_summary(self, input_size):
        summary(self, input_size=input_size)

    def model_visualize(self, device, loader, png_name, format="png"):
        batch = next(iter(loader))
        self.eval()
        yhat = self(batch[0].to(device))  # Give dummy batch to forward().

        make_dot(yhat, params=dict(list(self.named_parameters()))).render(
            png_name, format=format
        )
