import os
import argparse
import pickle

import yaml
import torch
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import data
import utils
from models import EffectRegressorMLP

parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
opts["device"] = "cpu"

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
paired = data.PairedObjectData(transform=transform)
paired.train = False

model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.decoder2.eval()

category = torch.load(os.path.join(opts["save"], "category.pt"))
if "discrete" in opts and not opts["discrete"]:
    e1 = torch.load(os.path.join(opts["save"], "centroids_1.pth"))
    e2 = torch.load(os.path.join(opts["save"], "centroids_2.pth"))
    codes = []
    for c in category:
        cc = torch.cat([e1[utils.binary_to_decimal(c[:2])],
                        e1[utils.binary_to_decimal(c[2:4])],
                        e2[utils.binary_to_decimal(c[4:])]])
        codes.append(cc)
    codes = torch.stack(codes)
    with torch.no_grad():
        e2_c = model.decoder2(codes)
else:
    with torch.no_grad():
        e2_c = model.decoder2(category.float())

_, label = torch.cdist(paired.effect, e2_c).topk(k=1, largest=False)
label = label.reshape(-1)

tree = DecisionTreeClassifier()
tree.fit(category, label)
preds = tree.predict(category)
print((torch.tensor(preds) == label).sum().float() / len(label))

file = open(os.path.join(opts["save"], "tree_nc.pkl"), "wb")
pickle.dump(tree, file)
file.close()

# cluster decoder effects for compactness
K = 5
ok = False
while not ok:
    ok = True
    clusters, _, _, _ = utils.kmeans(e2_c, k=K)
    clusters_un = clusters * paired.eff_std + paired.eff_mu
    effect_names = []
    for i, c_i in enumerate(clusters_un):
        print("Centroid %d: %.2f, %.2f, %.3f, %.2f, %.2f, %.3f" % ((i, ) + tuple(c_i)))

    for i, c_i in enumerate(clusters_un):
        print("Centroid %d: %.2f, %.2f, %.3f, %.2f, %.2f, %.3f" % ((i, ) + tuple(c_i)))
        print("Name the effect:")
        print(">>>", end="")
        name = input()
        if name == "reset":
            ok = False
            break
        effect_names.append(name)
effect_names = np.array(effect_names)
np.save(os.path.join(opts["save"], "effect_names.npy"), effect_names)

_, label = torch.cdist(paired.effect, clusters).topk(k=1, largest=False)
label = label.reshape(-1)

tree = DecisionTreeClassifier()
tree.fit(category, label)
preds = tree.predict(category)
print("%.1f" % ((torch.tensor(preds) == label).sum().float() / len(label)*100))

file = open(os.path.join(opts["save"], "tree.pkl"), "wb")
pickle.dump(tree, file)
file.close()
