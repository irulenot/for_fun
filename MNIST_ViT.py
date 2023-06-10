import torch
from tqdm import tqdm
import json
import importlib
import matplotlib.pyplot as plt
import numpy as np
import os

def main(json_data, seed_value, approach):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.manual_seed(seed_value)

    module = importlib.import_module(json_data['prepare_data']['module_name'])
    prepare_data = getattr(module, json_data['prepare_data']['function_name'])
    train_loader, test_loader = prepare_data()

    module = importlib.import_module(json_data['prepare_model']['module_name'])
    prepare_model = getattr(module, json_data['prepare_model']['function_name'])
    model, optimizer, train_function, test_function = prepare_model(device)

    num_epochs = json_data['epochs']
    train_losses = []
    test_losses = []
    test_accuracies = []
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss = train_function(model, optimizer, train_loader, device, epoch)
        test_loss, test_accuracy = test_function(model,test_loader, device)

        if epoch % 10 == 0 or epoch == num_epochs:
            torch.save(model.state_dict(), 'weights/' + approach + '_' + str(seed_value) + '_' + str(epoch) + '.pt')
        train_losses.extend(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy.cpu().item())

    # Load File
    metrics_path = 'metrics/metrics.json'
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as file:
            metrics = json.load(file)
    else:
        metrics = {}
    # Prepare json dict
    if approach not in metrics.keys():
        metrics[approach] = {}
    if seed_value not in metrics[approach].keys():
        metrics[approach][seed_value] = {}
    # Dump json dict
    metrics[approach][seed_value]['train_losses'] = train_losses
    metrics[approach][seed_value]['test_losses'] = test_losses
    metrics[approach][seed_value]['test_accuracies'] = test_accuracies
    with open(metrics_path, 'w+') as f:
        json.dump(metrics, f)

    return test_accuracies

if __name__ == "__main__":
    # seeds = [1, 2, 3, 4, 5]
    seeds = [2, 3, 4, 5]
    # seeds = [42]

    config_path_directed = 'config/MNIST_ViT.json'
    with open(config_path_directed, 'r') as file:
        json_data = json.load(file)
    config_path_directed = 'config/MNIST_ViT_directed.json'
    with open(config_path_directed, 'r') as file:
        json_data_directed = json.load(file)
    config_path_directed = 'config/MNIST_ViT_directed_first.json'
    with open(config_path_directed, 'r') as file:
        json_data_directed_first = json.load(file)
    config_path_directed = 'config/MNIST_ViT_directed_half.json'
    with open(config_path_directed, 'r') as file:
        json_data_directed_half = json.load(file)

    # all_accuracies = []
    # for seed in seeds:
    #     accuracies = main(json_data, seed, 'vanilla')
    #     all_accuracies.append(accuracies)
    # arr = np.array(all_accuracies)
    # column_averages = np.round(np.mean(arr, axis=0), 2)
    # averages = column_averages[::10]
    # averages = np.append(averages, column_averages[-1])
    # plt.plot(range(len(averages)), averages)
    # for i, j in zip(range(len(averages)), averages):
    #     plt.text(i, j, str(j), ha='center', va='bottom')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Test Accuracy')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('figures/ViT_vanilla.png')
    # plt.clf()

    # all_accuracies = []
    # for seed in seeds:
    #     accuracies = main(json_data_directed, seed, 'directed')
    #     all_accuracies.append(accuracies)
    # arr = np.array(all_accuracies)
    # column_averages = np.round(np.mean(arr, axis=0), 2)
    # averages = column_averages[::10]
    # averages = np.append(averages, column_averages[-1])
    # plt.plot(range(len(averages)), averages)
    # for i, j in zip(range(len(averages)), averages):
    #     plt.text(i, j, str(j), ha='center', va='bottom')
    # plt.xlabel('Epochs')
    # plt.ylabel('Accuracy')
    # plt.title('Test Accuracy')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig('figures/ViT_directed.png')
    # plt.clf()

    all_accuracies = []
    for seed in seeds:
        accuracies = main(json_data_directed_first, seed, 'directed')
        all_accuracies.append(accuracies)
    arr = np.array(all_accuracies)
    column_averages = np.round(np.mean(arr, axis=0), 2)
    averages = column_averages[::10]
    averages = np.append(averages, column_averages[-1])
    plt.plot(range(len(averages)), averages)
    for i, j in zip(range(len(averages)), averages):
        plt.text(i, j, str(j), ha='center', va='bottom')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('figures/ViT_directed_half.png')
    plt.clf()

    # all_accuracies = []
    # for seed in seeds:
    #     accuracies = main(json_data_directed_half, seed, 'directed')
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
    # plt.tight_layout()
    # plt.savefig('figures/ViT_directed_half.png')
    # plt.clf()