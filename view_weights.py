import numpy as np
import torch
import os
import glob


def main(approach):
    directory = 'weights/'
    file_list = glob.glob(directory + '/*' + approach + '*E5*')
    global_means, global_stds = [], []
    mean_layers = {'fc1.weight': [], 'fc1.bias': [], 'fc2.weight': [], 'fc2.bias': [], 'fc3.weight': [], 'fc3.bias': []}
    std_layers = {'fc1.weight': [], 'fc1.bias': [], 'fc2.weight': [], 'fc2.bias': [], 'fc3.weight': [], 'fc3.bias': []}
    for file in file_list:
        state_dict = torch.load(file)
        for param_tensor in state_dict:
            param = np.abs(state_dict[param_tensor].cpu().numpy().flatten())
            mean, std = np.mean(param), np.std(param)
            mean_layers[param_tensor].append(mean)
            std_layers[param_tensor].append(std)
            global_means.append(mean)
            global_stds.append(std)

    print(approach)
    for key in mean_layers.keys():
        print(key, "Mean:", np.mean(mean_layers[key]), "Standard Deviation:", np.mean(std_layers[key]))
    mean_value = np.mean(global_means)
    std_value = np.mean(global_stds)
    print("Total Mean:", mean_value)
    print("Total Standard Deviation:", std_value)
    print()

if __name__ == "__main__":
    main('vanilla')
    main('directed')
