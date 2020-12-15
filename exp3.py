import os
import argparse
import yaml
from sklearn.tree import DecisionTreeClassifier
import torch
import numpy as np

parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))

save_name = os.path.join(opts["save"], "domain.pddl")
if os.path.exists(save_name):
    os.remove(save_name)

category = torch.load(os.path.join(opts["save"], "category.pt"))
label = torch.load(os.path.join(opts["save"], "label.pt"))
effect_names = np.load(os.path.join(opts["save"], "effect_names.npy"))
K = len(effect_names)

tree = DecisionTreeClassifier()
tree.fit(category, label)
preds = tree.predict(category)
print((torch.tensor(preds) == label).sum().float() / len(label))
