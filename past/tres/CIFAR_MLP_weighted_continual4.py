import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(42)

# Define MLP model
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(3072, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# Load CIFAR-10 dataset
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = datasets.CIFAR10(root='./data', train=True, transform=transform, download=True)
test_dataset = datasets.CIFAR10(root='./data', train=False, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Initialize model and optimizer
model = MLP().to(device)
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()


# Training loop
def train(model, optimizer, criterion, train_loader, device, epoch):
    model.train()
    train_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)
        loss = criterion(output, target)

        loss.backward()

        percentile = 50 - ((epoch-1) * 20)
        if percentile > 0:
            with torch.no_grad():
                all_gradients = []
                for param in model.parameters():
                    if param.grad is not None:
                        all_gradients.append(param.grad.view(-1))

                all_gradients = torch.cat(all_gradients)
                gradients_abs = torch.abs(all_gradients)
                percentile = np.percentile(gradients_abs.cpu().numpy(), 50)

                for param in model.parameters():
                    if param.grad is not None:
                        top_50_percentile = torch.abs(param) >= percentile
                        param.grad[top_50_percentile] *= 2  # Double the gradients of top 50%
                        # param.grad[~top_50_percentile] -= torch.abs(param.grad[~top_50_percentile])  # Halve the gradients of bottom 50%

        optimizer.step()

        train_losses.append(loss.item())

        if batch_idx % 100 == 0:
            print('Train Batch: {} Loss: {:.6f}'.format(
                batch_idx, loss.item()))

    return train_losses


# Test loop
def test(model, criterion, test_loader, device):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset), accuracy))

    return test_loss, accuracy


# Training and testing
num_epochs = 5
train_losses = []
test_losses = []
test_accuracies = []

for epoch in range(1, num_epochs + 1):
    train_loss = train(model, optimizer, criterion, train_loader, device, epoch)
    test_loss, test_accuracy = test(model, criterion, test_loader, device)

    train_losses.extend(train_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# Apply moving average to smooth the metrics
window_size = 10

train_losses_smooth = np.convolve(train_losses, np.ones(window_size) / window_size, mode='valid')
test_losses_smooth = np.convolve(test_losses, np.ones(window_size) / window_size, mode='valid')

# Plot losses and accuracy
plt.figure(figsize=(10, 5))
plt.plot(range(window_size, len(train_losses_smooth) + window_size), train_losses_smooth, label='Training Loss')
plt.plot(range(1, len(test_losses_smooth) + 1), test_losses_smooth, label='Test Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training and Test Loss')
plt.legend()
plt.savefig('loss4.png')  # Save the figure
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(window_size, len(test_accuracies) + window_size), test_accuracies)
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Test Accuracy')
plt.savefig('accuracy4.png')  # Save the figure
plt.show()
