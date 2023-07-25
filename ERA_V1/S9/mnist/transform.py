from torchvision import transforms


def get_transform(type):
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
