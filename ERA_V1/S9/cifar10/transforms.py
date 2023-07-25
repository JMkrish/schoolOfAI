from torchvision import transforms

import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_album_transform(type):
    """Get albumentations transformer for the given type

    Returns:
        Compose: Composed albumentations transformations
    """

    trs = []

    if type == "train":
        trs.append(
            A.HorizontalFlip(
                p=0.3,
            )
        )
        trs.append(
            A.ShiftScaleRotate(
                shift_limit=0.15,
                scale_limit=0.15,
                rotate_limit=10,
                p=0.3,
            )
        )
        trs.append(
            A.CoarseDropout(
                max_holes=1,
                max_height=16,
                max_width=16,
                min_holes=1,
                min_height=16,
                min_width=16,
                fill_value=(0.4914, 0.4822, 0.4465),
                mask_fill_value=None,
                p=0.3,
            )
        )

    trs.append(
        A.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2470, 0.2435, 0.2616],
        )
    )
    trs.append(ToTensorV2())

    return A.Compose(trs)


def convert_to_tensor(image):
    return ToTensorV2()(image=image)


def get_torch_transform(type):
    """Get Pytorch Transform function for given type

    Returns:
        Compose: Composed transformations
    """
    if type == "train":
        # Train data transformations
        random_rotation_degree = 5
        img_size = (32, 32)
        random_crop_percent = (0.85, 1.0)
        transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(img_size, random_crop_percent),
                transforms.RandomRotation(random_rotation_degree),
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
