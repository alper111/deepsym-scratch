import os
import argparse
import time

import yaml
import torch
import matplotlib.pyplot as plt

from models import EffectRegressorMLP
import data
import utils

parser = argparse.ArgumentParser("Train effect prediction models.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))
file = open(os.path.join(opts["save"], "opts.yaml"), "w")
yaml.dump(opts, file)
file.close()
print(yaml.dump(opts))

device = torch.device(opts["device"])

# load the first level data
transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size1"], shuffle=True)

model = EffectRegressorMLP(opts)
if opts["load"] is not None:
    model.load(opts["load"], ext="", level=1)
    model.load(opts["load"], ext="", level=2)
model.print_model(1)
model.train(opts["epoch1"], loader, 1)

# load the best encoder1
model.load(opts["save"], "_best", 1)

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=2400, shuffle=False)
sample = iter(loader).next()
objects = sample["observation"].reshape(5, 10, 3, 4, 4, opts["size"], opts["size"])
objects = objects[:, :, 0].reshape(-1, 1, opts["size"], opts["size"])
model.encoder1.eval()
with torch.no_grad():
    codes = model.encoder1(objects.to(opts["device"])).cpu()

centroids, assigns, _, _ = utils.kmeans(codes, k=2**opts["code1_dim"])
plt.scatter(codes[:, 0], codes[:, 1], color="r", alpha=0.2)
plt.scatter(centroids[:, 0], centroids[:, 1], color="b", marker="x")
plt.show()

# codes = codes.reshape(5, 160, opts["code1_dim"])
assigns = assigns.reshape(5, 160)
code_table = torch.zeros(5, 2**opts["code1_dim"])
for i in range(5):
    for j in range(160):
        code_table[i][assigns[i, j]] += 1
print(code_table, file=open(os.path.join(opts["save"], "results.txt"), "w"))
