import torch

x = torch.tensor([
    [
        [160, 0, 0, 0],
        [0, 156, 1, 3],
        [0, 160, 0, 0],
        [9, 2, 142, 7],
        [0, 0, 0, 160]
    ],
    [
        [160, 0, 0, 0],
        [2, 158, 0, 0],
        [0, 160, 0, 0],
        [1, 0, 156, 3],
        [0, 0, 0, 160]
    ],
    [
        [158, 1, 0, 1],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [0, 1, 158, 1],
        [0, 0, 0, 160],
    ],
    [
        [160, 0, 0, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [0, 4, 156, 0],
        [0, 0, 0, 160]
    ],
    [
        [157, 0, 3, 0],
        [0, 154, 4, 2],
        [0, 154, 3, 3],
        [12, 0, 108, 40],
        [0, 160, 0, 0]
    ],
    [
        [160, 0, 0, 0],
        [0, 118, 41, 1],
        [1, 117, 41, 1],
        [154, 0, 6, 0],
        [0, 0, 0, 160]
    ],
    [
        [159, 0, 1, 0],
        [1, 151, 0, 8],
        [0, 159, 0, 1],
        [2, 0, 157, 1],
        [0, 0, 2, 158]
    ],
    [
        [158, 0, 2, 0],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [0, 0, 147, 13],
        [0, 0, 0, 160]
    ],
    [
        [160, 0, 0, 0],
        [0, 159, 0, 1],
        [0, 160, 0, 0],
        [0, 0, 159, 1],
        [0, 0, 0, 160]
    ],
    [
        [160, 0, 0, 0],
        [0, 157, 3, 0],
        [0, 157, 2, 1],
        [143, 0, 1, 16],
        [2, 5, 0, 153]
    ]
], dtype=torch.float)

x /= 160

mu = x.mean(dim=0)*100
std = x.std(dim=0)*100

for i in range(5):
    for j in range(4):
        print("%.1f $\\pm$ %.1f &" % (mu[i, j], std[i, j]), end=" ")
    print("\\\\")
