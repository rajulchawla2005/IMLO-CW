import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

# get training data
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

device = torch.device("cpu")
batch_size = 64
dataloader = DataLoader(training_data, batch_size=batch_size)


if __name__ == "__main__":
    # setup model
    # train model
    # store into .pth file
    for X, y in dataloader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    