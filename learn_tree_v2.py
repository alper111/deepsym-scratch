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
from utils import decimal_to_binary, binary_to_decimal_tensor

parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
opts["device"] = "cpu"

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)
trainset.train = False
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
sample = iter(loader).next()
with torch.no_grad():
    codes = model.concat(sample)
    dist = torch.distributions.Bernoulli(codes)
    for i in range(20):
        X.append(codes.round().clone())
    for i in range(20):
        sampled_codes = dist.sample()
        effects = binary_to_decimal_tensor(sampled_codes[:, :5])
        Y.append(effects)

X = torch.cat(X, dim=0)
Y = torch.cat(Y, dim=0)
# torch.save(X, "tempX.pt")
# torch.save(Y, "tempY.pt")

tree = DecisionTreeClassifier(min_samples_leaf=100)
tree.fit(X, Y)
preds = tree.predict(X)

print(tree.get_depth())
print(tree.get_n_leaves())
print(tree.get_params())
for v in tree.tree_.value:
    print(v)

accuracy = (torch.tensor(preds) == Y).sum().float() / len(Y)*100
print("%.1f" % accuracy)

path = os.path.join(opts["save"], "tree.pkl")
file = open(path, "wb")
pickle.dump(tree, file)
file.close()

unique_effect_categories = torch.unique(Y)
effect_names = []
for i, effect_cat in enumerate(unique_effect_categories):
    code = torch.tensor(decimal_to_binary(effect_cat, length=5))
    action = torch.tensor([1.0])
    z = torch.cat([code, action])
    with torch.no_grad():
        effect = model.decode(z)
    # print("%.3f %.3f %.3f %.3f %.3f %.3f" % tuple(effect), "===", "%.3f %.3f %.3f %.3f %.3f %.3f" % tuple(effect*std+mu), end="")
    if effect[0] < -0.3 and effect[1] < -0.1 and effect[2] > -0.11:
        effect_names.append("stack%d" % i)
        # print()
        # print(z)
        # print(" -> stacked")
    elif effect[0] < -0.3 and effect[1] < -0.1 and effect[2] < -0.11:
        effect_names.append("insert%d" % i)
        # print()
        # print(z)
        # print(" -> inserted")
    else:
        effect_names.append("effect%d" % i)
        # print()
        # print(z)
        # print(" -> unrelated")

path = os.path.join(opts["save"], "effect_names.npy")
np.save(path, effect_names)
