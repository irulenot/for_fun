import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json
import argparse
import importlib

def main(json_data):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_value = 42
    torch.manual_seed(seed_value)

    module = importlib.import_module(json_data['prepare_data']['module_name'])
    prepare_data = getattr(module, json_data['prepare_data']['function_name'])
    train_loader, test_loader = prepare_data()

    module = importlib.import_module(json_data['prepare_model']['module_name'])
    prepare_model = getattr(module, json_data['prepare_model']['function_name'])
    model, optimizer, criterion, train_function, test_function = prepare_model(device)

    num_epochs = json_data['epochs']
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss = train_function(model, optimizer, criterion, train_loader, device, epoch)
        test_loss, test_accuracy = test_function(model, criterion, test_loader, device)

        train_losses.extend(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    plt.plot(range(len(test_accuracies)), test_accuracies)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.savefig(json_data['figure_path'])
    plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-c', '--config_path', default='config/CIFAR_MLP.json')
    # args = parser.parse_args()
    config_path = 'config/CIFAR_MLP.json'
    with open(config_path, 'r') as file:
        json_data = json.load(file)
    main(json_data)