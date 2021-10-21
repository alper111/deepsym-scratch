import os
import argparse
import pickle

import yaml
import torch
from sklearn.tree import DecisionTreeClassifier
import numpy as np

import data
from models import DeepSymbolGenerator
from blocks import build_encoder, MLP, ChannelWrapper
from utils import decimal_to_binary, binary_to_decimal

parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
opts["device"] = "cpu"

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=2500, shuffle=True)
mu = trainset.eff_mu.numpy()
std = trainset.eff_std.numpy()

encoder1 = build_encoder(opts, 1).to(opts["device"])
decoder1 = MLP([opts["code1_dim"]+3] + [opts["hidden_dim"]]*opts["depth"] + [3]).to(opts["device"])
submodel = DeepSymbolGenerator(encoder1, decoder1, [], opts["device"], opts["learning_rate1"], os.path.join(opts["save"], "1"))
submodel.load("_best")
submodel.encoder = ChannelWrapper(submodel.encoder)
submodel.eval_mode()

encoder2 = build_encoder(opts, 2).to(opts["device"])
decoder2 = MLP([opts["code2_dim"]+opts["code1_dim"]*2+1] + [opts["hidden_dim"]]*opts["depth"] + [6]).to(opts["device"])
model = DeepSymbolGenerator(encoder2, decoder2, [submodel], opts["device"], opts["learning_rate2"], os.path.join(opts["save"], "2"))
model.load("_best")
model.eval_mode()
model.print_model()

X, Y = [], []
for i in range(10):
    sample = iter(loader).next()
    with torch.no_grad():
        codes = model.concat(sample).round()
        effects = []
        for c in codes:
            number = binary_to_decimal(c[:5])
            effects.append(number)

    X.append(codes)
    Y.append(torch.tensor(effects))

X = torch.cat(X, dim=0)
Y = torch.cat(Y, dim=0)
torch.save(X, "tempX.pt")
torch.save(Y, "tempY.pt")

tree = DecisionTreeClassifier(min_samples_leaf=500)
tree.fit(X, Y)
preds = tree.predict(X)

print(tree.get_depth())
print(tree.get_n_leaves())
print(tree.get_params())
# print(((torch.tensor(preds) - Y)**2).mean().float())
for v in tree.tree_.value:
    print(v)

accuracy = (torch.tensor(preds) == Y).sum().float() / len(Y)*100
print("%.1f" % accuracy)
# print("decoded values")
# for i, v in enumerate(Y):
#     print("Effect %d: %.2f %.2f %.2f %.2f %.2f %.2f" % ((i,) + tuple(v)))

path = os.path.join(opts["save"], "tree.pkl")
file = open(path, "wb")
pickle.dump(tree, file)
file.close()
