import numpy as np
import torch
from torchvision import datasets
from torch.utils.data import Dataset, DataLoader


class Album_Dataset(Dataset):
    def __init__(self, dataset, transforms=None):
        self.transforms = transforms
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Read Image and Label
        image, label = self.dataset[idx]

        image = np.array(image)

        # Apply Transforms
        if self.transforms is not None:
            image = self.transforms(image=image)["image"]

        return (image, label)


def get_dataset(train_transform, test_transform, kwargs):
    train_loader = DataLoader(
        WyDataset(
            datasets.CIFAR10("./data", train=True, download=True),
            transforms=train_transform,
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    test_loader = DataLoader(
        WyDataset(
            datasets.CIFAR10("./data", train=False, download=True),
            transforms=test_transform,
        ),
        batch_size=batch_size,
        shuffle=True,
        **kwargs
    )

    return train_loader, test_loader


def get_album_dataset(train_transform, test_transform, kwargs):
    train_loader = DataLoader(
        Album_Dataset(
            datasets.CIFAR10("./data", train=True, download=True),
            transforms=train_transform,
        ),
        **kwargs
    )

    test_loader = DataLoader(
        Album_Dataset(
            datasets.CIFAR10("./data", train=False, download=True),
            transforms=test_transform,
        ),
        **kwargs
    )

    return train_loader, test_loader
