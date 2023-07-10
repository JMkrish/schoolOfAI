import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
import matplotlib.pyplot as plt
from torchsummary import summary

from utils import is_cuda


def GetCorrectPredCount(pPrediction, pLabels):
    return pPrediction.argmax(dim=1).eq(pLabels).sum().item()


class MyNet(nn.Module):
    # This defines the structure of the NN.
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3)
        self.fc1 = nn.Linear(4096, 50)
        self.fc2 = nn.Linear(50, 10)
        self.criterion = nn.CrossEntropyLoss()

        # Data to plot accuracy and loss graphs
        self.train_losses = []
        self.test_losses = []
        self.train_acc = []
        self.test_acc = []

    def to(self, device):
        # Move all model parameters to the specified device
        self.device = device
        return super(MyNet, self).to(device)

    def set_criterion(self, criterion):
        self.criterion = criterion

    def forward(self, x):
        x = F.relu(self.conv1(x), 2)
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(self.conv3(x), 2)
        x = F.relu(F.max_pool2d(self.conv4(x), 2))
        x = x.view(-1, 4096)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

    def summary(self, size=(1, 28, 28)):
        summary(self, input_size=size)

    def train_model(self, train_loader, optimizer):
        super(MyNet, self).train()
        pbar = tqdm(train_loader)

        train_loss = 0
        correct = 0
        processed = 0

        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()

            # Predict
            pred = self(data)

            # Calculate loss
            loss = self.criterion(pred, target)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

            correct += GetCorrectPredCount(pred, target)
            processed += len(data)

            pbar.set_description(
                desc=f"Train: Loss={loss.item():0.4f} Batch_id={batch_idx} Accuracy={100*correct/processed:0.2f}"
            )

        self.train_acc.append(100 * correct / processed)
        self.train_losses.append(train_loss / len(train_loader))

    def test_model(self, test_loader):
        super(MyNet, self).eval()

        test_loss = 0
        correct = 0

        with torch.no_grad():
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(self.device), target.to(self.device)

                output = self(data)
                test_loss += self.criterion(output, target).item()  # sum up batch loss

                correct += GetCorrectPredCount(output, target)

        test_loss /= len(test_loader.dataset)
        self.test_acc.append(100.0 * correct / len(test_loader.dataset))
        self.test_losses.append(test_loss)

        print(
            "Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
                test_loss,
                correct,
                len(test_loader.dataset),
                100.0 * correct / len(test_loader.dataset),
            )
        )

    def my_plots(self, w, l, size):
        fig, axs = plt.subplots(w, l, figsize=size)
        axs[0, 0].plot(self.train_losses)
        axs[0, 0].set_title("Training Loss")
        axs[1, 0].plot(self.train_acc)
        axs[1, 0].set_title("Training Accuracy")
        axs[0, 1].plot(self.test_losses)
        axs[0, 1].set_title("Test Loss")
        axs[1, 1].plot(self.test_acc)
        axs[1, 1].set_title("Test Accuracy")
