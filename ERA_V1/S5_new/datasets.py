from torchvision import datasets


def get_minst_dataset(type, transform):
    if type == "train":
        return datasets.MNIST("../data", train=True, download=True, transform=transform)
    else:
        return datasets.MNIST(
            "../data", train=False, download=True, transform=transform
        )
