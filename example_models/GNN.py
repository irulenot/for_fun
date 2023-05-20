# SOURCE: https://github.com/DebasmitaGhose/PyTorch_Graph_Neural_Network_MNIST/blob/master/gnn_mnist.py

import numpy as np
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist


class GraphNet(nn.Module):
    def __init__(self, image_size=28, pred_edge=False):
        super(GraphNet, self).__init__()
        self.pred_edge = pred_edge
        N = image_size ** 2  # Number of pixels in the image
        self.fc = nn.Linear(N, 10, bias=False)
        # Create the adjacency matrix of size (N X N)
        if pred_edge:
            # Learn the adjacency matrix (learn to predict the edge between any pair of pixels)
            col, row = np.meshgrid(np.arange(image_size), np.arange(
                image_size))  # (28 x 28) Explanation: https://www.geeksforgeeks.org/numpy-meshgrid-function/
            coord = np.stack((col, row), axis=2).reshape(-1, 2)  # (784 x 2)
            coord_normalized = (coord - np.mean(coord, axis=0)) / (np.std(coord, axis=0) + 1e-5)  # Normalize the matrix
            coord_normalized = torch.from_numpy(coord_normalized).float()  # (784 x 2)
            adjacency_matrix = torch.cat((coord_normalized.unsqueeze(0).repeat(N, 1, 1),
                                          coord_normalized.unsqueeze(1).repeat(1, N, 1)), dim=2)  # (784 x 784 x 4)
            self.pred_edge_fc = nn.Sequential(nn.Linear(4, 64),
                                              nn.ReLU(),
                                              nn.Linear(64, 1),
                                              nn.Tanh())
            self.register_buffer('adjacency_matrix',
                                 adjacency_matrix)  # not to be considered a model paramater that is updated during training
        else:
            # Use a pre-computed adjacency matrix
            A = self.precompute_adjacency_images(image_size)
            self.register_buffer('A', A)  # not to be considered a model paramater that is updated during training

    def forward(self, x):
        '''
        x: image (batch_size x 1 x image_width x image_height)
        '''
        B = x.size(0)  # 64
        if self.pred_edge:
            self.A = self.pred_edge_fc(self.adjacency_matrix).squeeze()  # (784 x 784) --> predicted edge map

        avg_neighbor_features = (torch.bmm(self.A.unsqueeze(0).expand(B, -1, -1),
                                           x.view(B, -1, 1)).view(B, -1))  # (64 X 784)
        return self.fc(avg_neighbor_features)

    @staticmethod
    # Static method knows nothing about the class and just deals with the parameters.
    def precompute_adjacency_images(image_size):
        print('precompute_adjacency_images')
        col, row = np.meshgrid(np.arange(image_size), np.arange(
            image_size))  # (28 x 28) Explanation: https://www.geeksforgeeks.org/numpy-meshgrid-function/
        coord = np.stack((col, row), axis=2).reshape(-1, 2) / image_size  # (784 x 2) --> normalize
        dist = cdist(coord, coord)  # compute distance between every pair of pixels
        sigma = 0.05 * np.pi  # width of the Gaussian (can be a hyperparameter while training a model)
        A = np.exp(-dist / sigma ** 2)  # adjacency matrix of spatial similarity
        A[A < 0.01] = 0  # suppress values less than 0.01
        A = torch.from_numpy(A).float()

        # Normalization as per (Kipf & Welling, ICLR 2017)
        D = A.sum(1)  # nodes degree (N,)
        D_hat = (D + 1e-5) ** (-0.5)
        A_hat = D_hat.view(-1, 1) * A * D_hat.view(1, -1)  # N,N

        # Some additional trick I found to be useful
        A_hat[A_hat > 0.0001] = A_hat[A_hat > 0.0001] - 0.2

        print(A_hat[:10, :10])
        return A_hat