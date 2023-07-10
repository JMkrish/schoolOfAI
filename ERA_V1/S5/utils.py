import torch
import torch.nn.functional as F
import torch.optim as optim

from torchvision import datasets, transforms

import matplotlib.pyplot as plt
from torchsummary import summary


def is_cuda():
    cuda = torch.cuda.is_available()
    print("CUDA Available?", cuda)
    return cuda


def get_device():
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if is_cuda() else "cpu")
    return device


def get_optimizer(parameters, lr, momentum):
    return optim.SGD(parameters, lr=lr, momentum=momentum)


def get_optimizer(parameters, lr, momentum):
    return optim.SGD(parameters, lr=lr, momentum=momentum)


def get_scheduler(optimizer, step_size, gamma, verbose):
    return optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma, verbose=verbose
    )


def create_data_transforms():
    # Train data transformations
    transforms_train = transforms.Compose(
        [
            transforms.RandomApply(
                [
                    transforms.CenterCrop(22),
                ],
                p=0.1,
            ),
            transforms.Resize((28, 28)),
            transforms.RandomRotation((-15.0, 15.0), fill=0),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ]
    )
    # Test data transformations
    transforms_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    return transforms_train, transforms_test


def get_MNIST_data(path, train_transforms, test_transforms):
    train_data = datasets.MNIST(
        path, train=True, download=True, transform=train_transforms
    )
    test_data = datasets.MNIST(
        path, train=False, download=True, transform=test_transforms
    )
    return train_data, test_data


def create_data_loaders(train_data, test_data, **kwargs):
    train_loader = torch.utils.data.DataLoader(train_data, **kwargs)
    test_loader = torch.utils.data.DataLoader(test_data, **kwargs)
    return train_loader, test_loader


def plotter(data, labels, h, w):
    for i in range(h * w):
        plt.subplot(h, w, i + 1)
        plt.tight_layout()
        plt.imshow(data[i].squeeze(0), cmap="gray")
        plt.title(labels[i].item())
        plt.xticks([])
        plt.yticks([])


def model_summary(model, input_size):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    summary(model, input_size)
