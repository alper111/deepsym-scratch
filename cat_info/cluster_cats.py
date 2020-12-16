import torch

x = torch.tensor([
    [
        [160, 0, 0, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [80, 0, 80, 0],
        [0, 0, 0, 160]
    ],
    [
        [160, 0, 0, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [0, 0, 160, 0],
        [0, 0, 0, 160]
    ],
    [
        [128, 32, 0, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [96, 0, 64, 0],
        [0, 0, 0, 160]
    ],
    [
        [160, 0, 0, 0],
        [0, 144, 0, 16],
        [0, 136, 0, 24],
        [10, 0, 86, 64],
        [0, 60, 0, 100]
    ],
    [
        [160, 0, 0, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [96, 0, 64, 0],
        [0, 0, 0, 160]
    ],
    [
        [127, 32, 1, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [96, 0, 64, 0],
        [0, 0, 0, 160]
    ],
    [
        [160, 0, 0, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [0, 0, 160, 0],
        [0, 0, 0, 160]
    ],
    [
        [160, 0, 0, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [0, 0, 96, 64],
        [0, 160, 0, 0]
    ],
    [
        [160, 0, 0, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [0, 0, 160, 0],
        [0, 0, 0, 160]
    ],
    [
        [132, 0, 28, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [80, 0, 80, 0],
        [0, 0, 0, 160]
    ]
], dtype=torch.float)

x /= 160

mu = x.mean(dim=0)*100
std = x.std(dim=0)*100

for i in range(5):
    for j in range(4):
        print("%.1f $\\pm$ %.1f &" % (mu[i, j], std[i, j]), end=" ")
    print("\\\\")
