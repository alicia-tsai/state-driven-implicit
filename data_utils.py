"""
Helper functions for loading dataset.
"""

from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor, transforms


def load_fashion_mnist(download=False):
    train_data = datasets.FashionMNIST(
        root="data",
        train=True,
        download=download,
        transform=ToTensor(),
    )

    test_data = datasets.FashionMNIST(
        root="data",
        train=False,
        download=download,
        transform=ToTensor(),
    )
    return train_data, test_data


def load_cifar100(download=False):
    mean = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
    std = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    train_data = datasets.CIFAR100(
        root='data',
        train=True,
        download=download,
        transform=transform_train)

    test_data = datasets.CIFAR100(
        root="data",
        train=False,
        download=download,
        transform=transform_test
    )

    return train_data, test_data


def get_dataloader(train_data, test_data, batch_size=64):
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=10000)  # return full testing set

    for X, y in train_dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    return train_dataloader, test_dataloader
