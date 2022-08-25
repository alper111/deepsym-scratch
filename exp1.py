"""
Check the decision tree accuracy of DeepSym model.
"""
import os
import argparse
import time

import yaml
import torch
from sklearn.tree import DecisionTreeClassifier

from models import EffectRegressorMLP
import data

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

model.encoder1.eval()
model.decoder1.eval()
with torch.no_grad():
    preds = model.predict1(sample).cpu()

preds = preds * trainset.eff_std + trainset.eff_mu
truths = trainset.effect * trainset.eff_std + trainset.eff_mu
print((preds-truths).abs().mean(dim=0))

# buradaki effect clusteringe bir sey yapacak miyiz?!
assigns = torch.load("data/effect1_labels.pt")
with torch.no_grad():
    codes = model.encoder1(sample["observation"].to(opts["device"])).cpu()
category = torch.cat([codes, sample["action"]], dim=1)

tree = DecisionTreeClassifier()
tree.fit(category, assigns)
preds = tree.predict(category)
print((torch.tensor(preds) == assigns).sum().float() / len(assigns), file=open(os.path.join(opts["save"], "results.txt"), "w"))
