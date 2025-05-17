import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
# get the model architecture from train.py
import train

# get testing data
test_data = datasets.CIFAR10(
    root="data",
    train=False,
    download=True,
    transform=transforms.ToTensor()
)

DEVICE = torch.device("cpu")
BATCH_SIZE = 64
dataloader = DataLoader(test_data, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    # load model
    model = train.NeuralNetwork().to(DEVICE)
    model.load_state_dict(torch.load("model.pth", weights_only=True))

    # same loss function
    loss_fn = nn.CrossEntropyLoss()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    # testing mode
    model.eval()
    test_loss, correct = 0, 0
    # for every item in the test set
    with torch.no_grad():
        for X, y in dataloader:
            # put input and label on cpu
            X, y = X.to(DEVICE), y.to(DEVICE)
            # make a guess
            guess = model(X)
            # track loss and correctness
            test_loss += loss_fn(guess, y).item()
            correct += (guess.argmax(1) == y).type(torch.float).sum().item()
    # average 
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")