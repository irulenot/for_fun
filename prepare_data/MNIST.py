from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
import torchvision

def prepare_MNIST():
    DOWNLOAD_PATH = 'data/MNIST'
    BATCH_SIZE_TRAIN = 10000
    BATCH_SIZE_TEST = 1000

    transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                           transform=transform_mnist)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                          transform=transform_mnist)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)
    return train_loader, test_loader


def prepare_MNIST_small():
    DOWNLOAD_PATH = 'data/MNIST'
    BATCH_SIZE_TRAIN = 100
    BATCH_SIZE_TEST = 1000

    transform_mnist = torchvision.transforms.Compose([torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize((0.1307,), (0.3081,))])

    train_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=True, download=True,
                                           transform=transform_mnist)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE_TRAIN, shuffle=True)

    test_set = torchvision.datasets.MNIST(DOWNLOAD_PATH, train=False, download=True,
                                          transform=transform_mnist)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE_TEST, shuffle=True)
    return train_loader, test_loader