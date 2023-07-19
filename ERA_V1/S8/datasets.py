import torch
from torchvision import datasets


def get_minst_dataset(type, transform):
    if type == "train":
        return datasets.MNIST("../data", train=True, download=True, transform=transform)

    else:
        return datasets.MNIST(
            "../data", train=False, download=True, transform=transform
        )


def get_cifar10_dataset(type, transform):
    if type == "train":
        return datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        )
    else:
        return datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        )


# Pritns basic stats of the dataset
def print_dataset_bstats(dataset):
    # Create a DataLoader to iterate over the dataset
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset))

    # Iterate over the DataLoader to calculate the mean and standard deviation
    for images, _ in data_loader:
        mean = torch.mean(images, dim=(0, 2, 3))
        std = torch.std(images, dim=(0, 2, 3))
        break

    print(f"Data set size: {len(images)} elements")
    print(f"Image shape: {images[0].shape}")

    print("Mean:", mean)
    print("Standard Deviation:", std)
