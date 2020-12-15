import os
import argparse

import torch
import yaml

import data
import utils
from models import EffectRegressorMLP

parser = argparse.ArgumentParser("Save categories.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
opts["device"] = "cpu"
device = torch.device(opts["device"])

model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.encoder1.eval()
model.encoder2.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
X = torch.load("data/img/obs_prev_z.pt")
X = X.reshape(5, 10, 3, 4, 4, 42, 42)
X = X[:, :, 0, 2, 2]
X = X.reshape(-1, 1, 42, 42)
B, _, H, W = X.shape
Y = torch.empty(B, 1, opts["size"], opts["size"])

for i in range(B):
    Y[i] = transform(X[i])

with torch.no_grad():
    category1 = model.encoder1(Y.to(device))
if "discrete" in opts and not opts["discrete"]:
    centroids, indices, _, _ = utils.kmeans(category1, k=2**opts["code1_dim"])
    torch.save(centroids.cpu(), os.path.join(opts["save"], "centroids_1.pth"))
    category1 = torch.zeros(indices.shape[0], opts["code1_dim"])
    for i, idx in enumerate(indices):
        category1[i] = torch.tensor(utils.decimal_to_binary(idx, length=opts["code1_dim"]))
category1 = category1.int()

left_img = Y.repeat_interleave(B, 0)
right_img = Y.repeat(B, 1, 1, 1)
concat = torch.cat([left_img, right_img], dim=1)

category2 = model.encoder2(concat.to(device))
if "discrete" in opts and not opts["discrete"]:
    centroids, indices, _, _ = utils.kmeans(category2, k=2**opts["code2_dim"])
    torch.save(centroids.cpu(), os.path.join(opts["save"], "centroids_2.pth"))
    category2 = torch.zeros(indices.shape[0], opts["code2_dim"])
    for i, idx in enumerate(indices):
        category2[i] = torch.tensor(utils.decimal_to_binary(idx, length=opts["code2_dim"]))
category2 = category2.int()

left_cat = category1.repeat_interleave(B, 0)
right_cat = category1.repeat(B, 1)
category_all = torch.cat([left_cat, right_cat, category2], dim=-1)
torch.save(category_all.cpu(), os.path.join(opts["save"], "category.pt"))
