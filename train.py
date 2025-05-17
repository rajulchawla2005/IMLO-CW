import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchsummary import summary

# get training data
training_data = datasets.CIFAR10(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

train_set, val_set = torch.utils.data.random_split(training_data, [40000, 10000])

epochs = 5
device = torch.device("cpu")
batch_size = 64
train_dataloader = DataLoader(train_set, batch_size=batch_size)
test_dataloader = DataLoader(val_set, batch_size=batch_size)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.stack = nn.Sequential(
            # 3x32x32 -> 16x16x16
            nn.Conv2d(
                in_channels=3,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            # 16x16x16 -> 32x8x8
            nn.Conv2d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),

            nn.Flatten(),
            nn.Linear(in_features=32*8*8, 
                      out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, 
                      out_features=10)
        )

    def forward(self, x):
        return self.stack(x)

def train():
    size = len(train_dataloader.dataset)
    # training mode
    model.train()
    # for every item in the training set
    for batch, (X, y) in enumerate(train_dataloader):
        # put the input and label on the cpu
        X, y = X.to(device), y.to(device)
        # make a guess
        guess = model(X)
        # calculate a loss
        loss = loss_fn(guess, y)

        # backpropagation
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train_test():
    size = len(test_dataloader.dataset)
    num_batches = len(test_dataloader)
    # testing mode
    model.eval()
    test_loss, correct = 0, 0
    # for every item in the test set
    with torch.no_grad():
        for X, y in test_dataloader:
            # put input and label on cpu
            X, y = X.to(device), y.to(device)
            # make a guess
            guess = model(X)
            # track loss and correctness
            test_loss += loss_fn(guess, y).item()
            correct += (guess.argmax(1) == y).type(torch.float).sum().item()
    # average 
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

if __name__ == "__main__":
    # setup model
    # train model
    # store into .pth file

    model = NeuralNetwork().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimiser = torch.optim.SGD(model.parameters(), lr=0.01)

    for t in range(epochs):
        print(f"Epoch {t+1}\n")
        train()
        train_test()
    print("Done!")

    summary(model, input_size=(3, 32, 32))

    torch.save(model.state_dict(), "model.pth")
    print("Saved PyTorch Model State to model.pth")
