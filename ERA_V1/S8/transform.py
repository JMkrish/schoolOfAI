from torchvision import transforms


def get_mnist_transform(type):
    if type == "train":
        # Train data transformations
        transform = transforms.Compose(
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
    else:
        # Test data transformations
        transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        )

    return transform


def get_cifar10_transform(type):
    if type == "train":
        # Train data transformations
        transform = transforms.Compose(
            [
                transforms.RandomApply(
                    [
                        transforms.CenterCrop(25),
                    ],
                    p=0.1,
                ),
                transforms.Resize((32, 32)),
                transforms.RandomRotation((-15.0, 15.0), fill=0),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
                ),
            ]
        )
    else:
        # Test data transformations
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.4914, 0.4822, 0.4465], [0.2470, 0.2435, 0.2616]
                ),
            ]
        )

    return transform
