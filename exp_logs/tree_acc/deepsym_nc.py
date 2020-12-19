import torch

x = torch.tensor([
    31.2,
    42.7,
    44.3,
    44.0,
    42.1,
    44.0,
    40.8,
    40.2,
    45.8,
    45.6
])

print(x.mean())
print(x.std())
