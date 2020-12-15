import os
import argparse
import pickle

import yaml
import torch
from sklearn.tree import DecisionTreeClassifier

parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))

save_name = os.path.join(opts["save"], "domain.pddl")
if os.path.exists(save_name):
    os.remove(save_name)

category = torch.load(os.path.join(opts["save"], "category.pt"))
label = torch.load(os.path.join(opts["save"], "label.pt"))

tree = DecisionTreeClassifier()
tree.fit(category, label)
preds = tree.predict(category)
print((torch.tensor(preds) == label).sum().float() / len(label))

file = open(os.path.join(opts["save"], "tree.pkl"), "wb")
pickle.dump(tree, file)
file.close()
