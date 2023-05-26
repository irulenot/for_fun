# SOURCE: https://towardsdatascience.com/a-demonstration-of-using-vision-transformers-in-pytorch-mnist-handwritten-digit-recognition-407eafbc15b0

import time

import torch
import torch.nn.functional as F
import torch.optim as optim
import torchvision

from example_models.ViT import ViT
import numpy as np


def train_epoch(model, optimizer, data_loader, loss_history, epoch, not_pruned):
    total_samples = len(data_loader.dataset)
    model.train()

    for i, (data, target) in enumerate(data_loader):
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()

        percentile = 50
        if epoch <= 6:
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

        optimizer.step()

        if i % 100 == 0:
            print('[' +  '{:5}'.format(i * len(data)) + '/' + '{:5}'.format(total_samples) +
                  ' (' + '{:3.0f}'.format(100 * i / len(data_loader)) + '%)]  Loss: ' +
                  '{:6.4f}'.format(loss.item()))
            loss_history.append(loss.item())


def evaluate(model, data_loader, loss_history):
    model.eval()

    total_samples = len(data_loader.dataset)
    correct_samples = 0
    total_loss = 0

    with torch.no_grad():
        for data, target in data_loader:
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()

    avg_loss = total_loss / total_samples
    loss_history.append(avg_loss)
    print('\nAverage test loss: ' + '{:.4f}'.format(avg_loss) +
          '  Accuracy:' + '{:5}'.format(correct_samples) + '/' +
          '{:5}'.format(total_samples) + ' (' +
          '{:4.2f}'.format(100.0 * correct_samples / total_samples) + '%)\n')


def main():
    torch.manual_seed(42)

    DOWNLOAD_PATH = '../../data/MNIST'
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

    N_EPOCHS = 10

    start_time = time.time()
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=6, heads=8, mlp_dim=128)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    train_loss_history, test_loss_history = [], []
    not_pruned = True
    for epoch in range(1, N_EPOCHS + 1):
        print('Epoch:', epoch)
        train_epoch(model, optimizer, train_loader, train_loss_history, epoch, not_pruned)
        evaluate(model, test_loader, test_loss_history)

    print('Execution time:', '{:5.2f}'.format(time.time() - start_time), 'seconds')

if __name__ == "__main__":
    main()