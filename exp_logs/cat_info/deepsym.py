import torch

x = torch.tensor([
    [
        [160, 0, 0, 0],
        [0, 138, 10, 12],
        [3, 140, 14, 3],
        [78, 25, 55, 2],
        [0, 0, 0, 160]
    ],
    [
        [158, 2, 0, 0],
        [0, 160, 0, 0],
        [0, 155, 5, 0],
        [0, 2, 158, 0],
        [0, 0, 0, 160]
    ],
    [
        [160, 0, 0, 0],
        [0, 159, 0, 1],
        [0, 148, 0, 12],
        [1, 0, 139, 20],
        [0, 0, 0, 160]
    ],
    [
        [129, 2, 28, 1],
        [0, 153, 7, 0],
        [0, 159, 1, 0],
        [0, 30, 128, 2],
        [0, 0, 0, 160]
    ],
    [
        [160, 0, 0, 0],
        [7, 143, 10, 0],
        [24, 133, 3, 0],
        [127, 0, 33, 0],
        [0, 0, 0, 160]
    ],
    [
        [151, 0, 0, 9],
        [0, 160, 0, 0],
        [2, 150, 0, 8],
        [36, 3, 118, 3],
        [0, 0, 0, 160]
    ],
    [
        [131, 21, 0, 8],
        [0, 145, 0, 15],
        [0, 148, 0, 12],
        [0, 50, 110, 0],
        [1, 0, 0, 159]
    ],
    [
        [129, 3, 23, 5],
        [0, 160, 0, 0],
        [0, 160, 0, 0],
        [6, 8, 146, 0],
        [0, 0, 0, 160]
    ],
    [
        [148, 12, 0, 0],
        [15, 108, 28, 9],
        [1, 157, 1, 1],
        [3, 0, 157, 0],
        [0, 0, 0, 160]
    ],
    [
        [160, 0, 0, 0],
        [0, 160, 0, 0],
        [1, 147, 5, 7],
        [1, 3, 151, 5],
        [0, 0, 0, 160]
    ],
], dtype=torch.float)

x /= 160

mu = x.mean(dim=0)*100
std = x.std(dim=0)*100

for i in range(5):
    for j in range(4):
        print("%.1f $\\pm$ %.1f &" % (mu[i, j], std[i, j]), end=" ")
    print("\\\\")