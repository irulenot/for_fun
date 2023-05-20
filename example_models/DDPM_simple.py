# SOURCE: https://github.com/BrianPulfer/PapersReimplementations/blob/main/ddpm/models.py
# SOURCE: https://medium.com/mlearning-ai/enerating-images-with-ddpms-a-pytorch-implementation-cef5a2ba8cb1

import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_embedding(n, d):
    # Returns the standard positional embedding
    embedding = torch.zeros(n, d)
    wk = torch.tensor([1 / 10_000 ** (2 * j / d) for j in range(d)])
    wk = wk.reshape((1, d))
    t = torch.arange(n).reshape((n, 1))
    embedding[:,::2] = torch.sin(t * wk[:,::2])
    embedding[:,1::2] = torch.cos(t * wk[:,::2])

    return embedding


# DDPM class
class MyDDPM(nn.Module):
    def __init__(self, network, n_steps=200, min_beta=10 ** -4, max_beta=0.02, device=None, image_chw=(1, 28, 28)):
        super(MyDDPM, self).__init__()
        self.n_steps = n_steps
        self.device = device
        self.image_chw = image_chw
        self.network = network.to(device)
        self.betas = torch.linspace(min_beta, max_beta, n_steps).to(
            device)  # Number of steps is typically in the order of thousands
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.tensor([torch.prod(self.alphas[:i + 1]) for i in range(len(self.alphas))]).to(device)

    def forward(self, x0, t, eta=None):
        # Make input image more noisy (we can directly skip to the desired step)
        n, c, h, w = x0.shape
        a_bar = self.alpha_bars[t]

        if eta is None:
            eta = torch.randn(n, c, h, w).to(self.device)

        noisy = a_bar.sqrt().reshape(n, 1, 1, 1) * x0 + (1 - a_bar).sqrt().reshape(n, 1, 1, 1) * eta
        return noisy

    def backward(self, x, t):
        # Run each image through the network for each timestep t in the vector t.
        # The network returns its estimation of the noise that was added.
        return self.network(x, t)


class MyUNet(nn.Module):
    def __init__(self, input_dim, n_steps=1000, time_emb_dim=100):
        super(MyUNet, self).__init__()

        self.input_dim = input_dim

        # Sinusoidal embedding
        self.time_embed = nn.Embedding(n_steps, time_emb_dim)
        self.time_embed.weight.data = sinusoidal_embedding(n_steps, time_emb_dim)
        self.time_embed.requires_grad_(False)

        # First half
        self.fc1 = nn.Linear(self.input_dim, 512)
        self.te1 = self._make_te(time_emb_dim, 1)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.te2 = self._make_te(time_emb_dim, 1)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.te3 = self._make_te(time_emb_dim, 1)

        # Second half
        self.fc4 = nn.Linear(self.fc3.out_features, self.fc3.out_features * 2)
        self.te4 = self._make_te(time_emb_dim, 1)
        self.fc5 = nn.Linear(self.fc4.out_features, self.fc4.out_features * 2)
        self.te5 = self._make_te(time_emb_dim, 1)
        self.fc6 = nn.Linear(self.fc5.out_features, self.input_dim)
        self.te6 = self._make_te(time_emb_dim, 1)

    def forward(self, x, t):
        # x is (N, 2, 28, 28) (image with positional embedding stacked on channel dimension)
        t = self.time_embed(t)
        n = len(x)
        original_shape = x.shape
        x = x.reshape(original_shape[0], -1)
        x = F.leaky_relu(self.fc1(x + self.te1(t).reshape(n, -1)), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x + self.te2(t).reshape(n, -1)), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x + self.te3(t).reshape(n, -1)), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc4(x + self.te4(t).reshape(n, -1)), 0.2)
        x = F.leaky_relu(self.fc5(x + self.te5(t).reshape(n, -1)), 0.2)
        x = F.leaky_relu(self.fc6(x + self.te6(t).reshape(n, -1)), 0.2)
        x = torch.tanh(x)
        return x.reshape(original_shape)

    def _make_te(self, dim_in, dim_out):
        return nn.Sequential(
            nn.Linear(dim_in, dim_out),
            nn.SiLU(),
            nn.Linear(dim_out, dim_out)
        )