import torch
from tqdm import tqdm
import json
import importlib
import matplotlib.pyplot as plt
import numpy as np

def main(json_data, seed_value, approach):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    for epoch in range(1, num_epochs + 1):
        train_loss = train_function(model, optimizer, criterion, train_loader, device, epoch)
        test_loss, test_accuracy = test_function(model, criterion, test_loader, device)

        torch.save(model.state_dict(), 'weights/A' + approach + '_S' + str(seed_value) + '_E' + str(epoch) + '.pt')
        train_losses.extend(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

    return test_accuracies

if __name__ == "__main__":
    seeds = [42]

    config_path_directed = 'config/CIFAR_MLP_rapid.json'
    with open(config_path_directed, 'r') as file:
        json_data = json.load(file)
    config_path_directed = 'config/CIFAR_MLP_directed_rapid.json'
    with open(config_path_directed, 'r') as file:
        json_data_directed = json.load(file)
    config_path = 'config/CIFAR_MLP_directed_extreme_rapid.json'
    with open(config_path, 'r') as file:
        json_data_extreme = json.load(file)

    # all_accuracies = []
    # for seed in tqdm(seeds):
    #     accuracies = main(json_data, seed, 'vanilla')
    #     all_accuracies.append(accuracies)
    # arr = np.array(all_accuracies)
    # column_averages = np.round(np.mean(arr, axis=0), 2)
    # plt.plot(range(len(column_averages)), column_averages)
    # for i, j in zip(range(len(column_averages)), column_averages):
    #     plt.text(i, j, str(j), ha='center', va='bottom')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Test Accuracy')
    # plt.grid(True)
    # plt.savefig('figures/vanilla_single.png')
    # plt.clf()
    #
    # all_accuracies = []
    # for seed in tqdm(seeds):
    #     accuracies = main(json_data_directed, seed, 'directed')
    #     all_accuracies.append(accuracies)
    # arr = np.array(all_accuracies)
    # column_averages = np.round(np.mean(arr, axis=0), 2)
    # plt.plot(range(len(column_averages)), column_averages)
    # for i, j in zip(range(len(column_averages)), column_averages):
    #     plt.text(i, j, str(j), ha='center', va='bottom')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Test Accuracy')
    # plt.grid(True)
    # plt.savefig('figures/directed_single.png')
    # plt.clf()

    all_accuracies = []
    for seed in tqdm(seeds):
        accuracies = main(json_data_extreme, seed, 'extreme')
        all_accuracies.append(accuracies)
    arr = np.array(all_accuracies)
    column_averages = np.round(np.mean(arr, axis=0), 2)
    plt.plot(range(len(column_averages)), column_averages)
    for i, j in zip(range(len(column_averages)), column_averages):
        plt.text(i, j, str(j), ha='center', va='bottom')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.grid(True)
    plt.savefig('figures/directed_extreme_single.png')
    plt.clf()