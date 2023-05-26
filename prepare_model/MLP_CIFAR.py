import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

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

def train(model, optimizer, criterion, train_loader, device, epcoh=0):
    model.train()
    train_losses = []

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
        # if batch_idx % 100 == 0:
        #     print('Train Batch: {} Loss: {:.6f}'.format(
        #         batch_idx, loss.item()))
    return train_losses


def train_directed_extreme(model, optimizer, criterion, train_loader, device, epoch):
    model.train()
    train_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # Reward top 50% and penalize bottom 50% connections in the first epoch
        if epoch % 2 == 1:
            with torch.no_grad():
                all_gradients = []
                for param in model.parameters():
                    if param.grad is not None:
                        gradients_abs = torch.abs(param.grad.view(-1))
                        percentile_50 = np.percentile(gradients_abs.cpu().numpy(), 50)
                        all_gradients.append(percentile_50)

                for i, param in enumerate(model.parameters()):
                    if param.grad is not None:
                        top_50_percentile = torch.abs(param) >= all_gradients[i]
                        param.grad[top_50_percentile] *= 2
        optimizer.step()
        train_losses.append(loss.item())
        # if batch_idx % 100 == 0:
        #     print('Train Batch: {} Loss: {:.6f}'.format(
        #         batch_idx, loss.item()))
    return train_losses


def train_directed(model, optimizer, criterion, train_loader, device, epoch):
    model.train()
    train_losses = []
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # Reward top 50% and penalize bottom 50% connections in the first epoch
        if epoch == 1:
            with torch.no_grad():
                all_gradients = []
                for param in model.parameters():
                    if param.grad is not None:
                        all_gradients.append(param.grad.view(-1))

                all_gradients = torch.cat(all_gradients)
                gradients_abs = torch.abs(all_gradients)
                percentile_50 = np.percentile(gradients_abs.cpu().numpy(), 50)

                for param in model.parameters():
                    if param.grad is not None:
                        top_50_percentile = torch.abs(param) >= percentile_50
                        param.grad[top_50_percentile] *= 2
        optimizer.step()
        train_losses.append(loss.item())
        # if batch_idx % 100 == 0:
        #     print('Train Batch: {} Loss: {:.6f}'.format(
        #         batch_idx, loss.item()))
    return train_losses

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
    # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
    #     test_loss, correct, len(test_loader.dataset), accuracy))
    return test_loss, accuracy

def prepare_model(device):
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion, train, test

def prepare_model_directed(device):
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion, train_directed, test

def prepare_model_directed_extreme(device):
    model = MLP().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    return model, optimizer, criterion, train_directed_extreme, test