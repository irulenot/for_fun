# SOURCE: https://towardsdatascience.com/a-demonstration-of-using-vision-transformers-in-pytorch-mnist-handwritten-digit-recognition-407eafbc15b0

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
import torch.optim as optim
import numpy as np


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        self.to_qkv = nn.Linear(dim, dim * 3, bias=False)
        self.to_out = nn.Linear(dim, dim)

    def forward(self, x, mask=None):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=h)

        dots = torch.einsum('bhid,bhjd->bhij', q, k) * self.scale

        if mask is not None:
            mask = F.pad(mask.flatten(1), (1, 0), value=True)
            assert mask.shape[-1] == dots.shape[-1], 'mask has incorrect dimensions'
            mask = mask[:, None, :] * mask[:, :, None]
            dots.masked_fill_(~mask, float('-inf'))
            del mask

        attn = dots.softmax(dim=-1)

        out = torch.einsum('bhij,bhjd->bhid', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        return out


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, mlp_dim):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Residual(PreNorm(dim, Attention(dim, heads=heads))),
                Residual(PreNorm(dim, FeedForward(dim, mlp_dim)))
            ]))

    def forward(self, x, mask=None):
        for attn, ff in self.layers:
            x = attn(x, mask=mask)
            x = ff(x)
        return x


class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, channels=3):
        super().__init__()
        assert image_size % patch_size == 0, 'image dimensions must be divisible by the patch size'
        num_patches = (image_size // patch_size) ** 2
        patch_dim = channels * patch_size ** 2

        self.patch_size = patch_size

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.patch_to_embedding = nn.Linear(patch_dim, dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = Transformer(dim, depth, heads, mlp_dim)

        self.to_cls_token = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Linear(mlp_dim, num_classes)
        )

    def forward(self, img, mask=None):
        p = self.patch_size

        x = rearrange(img, 'b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=p, p2=p)
        x = self.patch_to_embedding(x)

        cls_tokens = self.cls_token.expand(img.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding
        x = self.transformer(x, mask)

        x = self.to_cls_token(x[:, 0])
        return self.mlp_head(x)

def train_epoch(model, optimizer, train_loader, device, epoch=0):
    model.train()
    train_losses = []
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses

def train_epoch_directed(model, optimizer, train_loader, device, epoch=0):
    model.train()
    train_losses = []
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        if epoch % 2 == 1:
            with torch.no_grad():
                all_gradients = []
                for name, param in model.named_parameters():
                    if name[:5] == 'trans' or name[:3] == 'mlp' or name[:5] == 'patch':
                        if param.grad is not None:
                            gradients_abs = torch.abs(param.grad.view(-1))
                            percentile_50 = np.percentile(gradients_abs.cpu().numpy(), 50)
                            all_gradients.append(percentile_50)
                i2 = 0
                for name_param in model.named_parameters():
                    name, param = name_param
                    if name[:5] == 'trans' or name[:3] == 'mlp':
                        if param.grad is not None:
                            top_50_percentile = torch.abs(param) >= all_gradients[i2]
                            param.grad[top_50_percentile] *= 2
                            i2 += 1
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses

def train_epoch_directed_first(model, optimizer, train_loader, device, epoch=0):
    model.train()
    train_losses = []
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        if epoch % 2 == 1 and epoch < 50:
            with torch.no_grad():
                all_gradients = []
                for name, param in model.named_parameters():
                    if name[:5] == 'trans' or name[:3] == 'mlp' or name[:5] == 'patch':
                        if param.grad is not None:
                            gradients_abs = torch.abs(param.grad.view(-1))
                            percentile_50 = np.percentile(gradients_abs.cpu().numpy(), 50)
                            all_gradients.append(percentile_50)
                i2 = 0
                for name_param in model.named_parameters():
                    name, param = name_param
                    if name[:5] == 'trans' or name[:3] == 'mlp':
                        if param.grad is not None:
                            top_50_percentile = torch.abs(param) >= all_gradients[i2]
                            param.grad[top_50_percentile] *= 2
                            i2 += 1
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses

def train_epoch_directed_half(model, optimizer, train_loader, device, epoch=0):
    model.train()
    train_losses = []
    for i, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = F.log_softmax(model(data), dim=1)
        loss = F.nll_loss(output, target)
        loss.backward()
        if epoch <= 1:
            with torch.no_grad():
                all_gradients = []
                for name, param in model.named_parameters():
                    if name[:5] == 'trans' or name[:3] == 'mlp' or name[:5] == 'patch':
                        if param.grad is not None:
                            gradients_abs = torch.abs(param.grad.view(-1))
                            percentile_50 = np.percentile(gradients_abs.cpu().numpy(), 50)
                            all_gradients.append(percentile_50)
                i2 = 0
                for name_param in model.named_parameters():
                    name, param = name_param
                    if name[:5] == 'trans' or name[:3] == 'mlp':
                        if param.grad is not None:
                            top_50_percentile = torch.abs(param) >= all_gradients[i2]
                            param.grad[top_50_percentile] *= 2
                            i2 += 1
        optimizer.step()
        train_losses.append(loss.item())
    return train_losses

def evaluate(model, data_loader, device):
    model.eval()
    correct_samples = 0
    total_samples = 0
    total_loss = 0
    test_losses = []
    with torch.no_grad():
        for data, target in data_loader:
            data, target = data.to(device), target.to(device)
            output = F.log_softmax(model(data), dim=1)
            loss = F.nll_loss(output, target, reduction='sum')
            _, pred = torch.max(output, dim=1)

            total_loss += loss.item()
            correct_samples += pred.eq(target).sum()
            total_samples += len(data)
            test_losses.append(loss.item())
    accuracy = correct_samples/total_samples*100
    return test_losses, accuracy

def prepare_model(device):
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=6, heads=8, mlp_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    return model, optimizer, train_epoch, evaluate

def prepare_model_directed(device):
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=6, heads=8, mlp_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    return model, optimizer, train_epoch_directed, evaluate

def prepare_model_directed_first(device):
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=6, heads=8, mlp_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    return model, optimizer, train_epoch_directed_first, evaluate

def prepare_model_directed_half(device):
    model = ViT(image_size=28, patch_size=7, num_classes=10, channels=1,
                dim=64, depth=6, heads=8, mlp_dim=128).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)

    return model, optimizer, train_epoch_directed_half, evaluate