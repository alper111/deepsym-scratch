import os

import torch
from torchvision import transforms
import numpy as np


class SingleObjectData(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.state = torch.load("data/img/obs_prev_z.pt").unsqueeze(1)
        self.action = torch.load("data/img/action.pt")

        self.effect = torch.load("data/img/delta_pix_1.pt")
        self.eff_mu = self.effect.mean(dim=0)
        self.eff_std = self.effect.std(dim=0) + 1e-6
        self.effect = (self.effect - self.eff_mu) / self.eff_std

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        sample = {}
        sample["state"] = self.state[idx]
        sample["effect"] = self.effect[idx]
        sample["action"] = self.action[idx]
        if self.transform:
            sample["state"] = self.transform(self.state[idx])
        return sample


class PairedObjectData(torch.utils.data.Dataset):
    def __init__(self, transform=None):
        self.transform = transform
        self.train = True
        self.state = torch.load("data/img/obs_prev_z.pt")
        self.state = self.state.reshape(5, 10, 3, 4, 4, 42, 42)
        self.state = self.state[:, :, 0]

        self.effect = torch.load("data/img/delta_pix_3.pt")
        self.effect = self.effect.abs()
        self.eff_mu = self.effect.mean(dim=0)
        self.eff_std = self.effect.std(dim=0) + 1e-6
        self.effect = (self.effect - self.eff_mu) / self.eff_std

    def __len__(self):
        return len(self.effect)

    def __getitem__(self, idx):
        sample = {}
        obj_i = idx // 500
        size_i = (idx // 50) % 10
        obj_j = (idx // 10) % 5
        size_j = idx % 10
        if self.train:
            ix = np.random.randint(0, 4)
            iy = np.random.randint(0, 4)
            jx = np.random.randint(0, 4)
            jy = np.random.randint(0, 4)
        else:
            ix, iy, jx, jy = 2, 2, 2, 2
        img_i = self.state[obj_i, size_i, ix, iy]
        img_j = self.state[obj_j, size_j, jx, jy]
        if self.transform:
            img_i = self.transform(img_i)
            img_j = self.transform(img_j)
            sample["state"] = torch.cat([img_i, img_j])
        else:
            sample["state"] = torch.stack([img_i, img_j])
        sample["effect"] = self.effect[idx]
        sample["action"] = torch.tensor([1.0])
        return sample


def default_transform(size, affine, mean=None, std=None):
    transform = [transforms.ToPILImage()]
    if size:
        transform.append(transforms.Resize(size))
    if affine:
        transform.append(
            transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                fillcolor=int(0.285*255)
            )
        )
    transform.append(transforms.ToTensor())
    if mean is not None:
        transform.append(transforms.Normalize([mean], [std]))
    transform = transforms.Compose(transform)
    return transform


class TilePuzzleData(torch.utils.data.Dataset):
    def __init__(self, path):
        self.state = torch.load(os.path.join(path, "tile_state.pt"))
        self.effect = torch.load(os.path.join(path, "tile_effect.pt"))
        self.action = torch.load(os.path.join(path, "tile_action.pt"))

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        sample = {}
        sample["state"] = self.state[idx] / 255.0
        sample["effect"] = self.effect[idx] / 255.0
        sample["action"] = self.action[idx].float()
        return sample
