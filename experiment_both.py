import torch
from tqdm import tqdm
import os
import json
import importlib

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
    for epoch in tqdm(range(1, num_epochs + 1)):
        train_loss = train_function(model, optimizer, criterion, train_loader, device, epoch)
        test_loss, test_accuracy = test_function(model, criterion, test_loader, device)

        torch.save(model.state_dict(), 'weights/A' + approach + '_S' + str(seed_value) + '_E' + str(epoch) + '.pt')
        train_losses.extend(train_loss)
        test_losses.append(test_loss)
        test_accuracies.append(test_accuracy)

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

if __name__ == "__main__":
    seeds = [1, 2, 3, 4, 5]

    config_path = 'config/CIFAR_MLP.json'
    with open(config_path, 'r') as file:
        json_data = json.load(file)

    config_path_directed = 'config/CIFAR_MLP_directed.json'
    with open(config_path_directed, 'r') as file:
        json_data_directed = json.load(file)

    for seed in seeds:
        main(json_data, seed, 'vanilla')
        main(json_data_directed, seed, 'directed')