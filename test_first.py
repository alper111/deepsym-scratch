import argparse
import os

import torch
import torchvision
import yaml
import matplotlib.pyplot as plt

import data
import utils
from models import DeepSymbolGenerator
from blocks import MLP, build_encoder

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
opts["device"] = "cpu"

encoder = build_encoder(opts, 1).to(opts["device"])
decoder = MLP([opts["code1_dim"]+3] + [opts["hidden_dim"]]*opts["depth"] + [3]).to(opts["device"])
model = DeepSymbolGenerator(encoder, decoder, [], opts["device"], opts["learning_rate1"], os.path.join(opts["save"], "1"))
model.load("_best")
model.eval_mode()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=2400, shuffle=True)
sample = iter(loader).next()
objects = sample["state"].reshape(5, 10, 3, 4, 4, opts["size"], opts["size"])
objects = objects[:, :, 0].reshape(-1, 1, 42, 42)
colored = [[], [], [], []]
NUM_OBJECTS = 10

with torch.no_grad():
    done = False
    it = 0
    while not done:
        c = model.encode(objects[it].reshape(1, 1, 42, 42)).round()
        cat = int(utils.binary_to_decimal(c[0]))
        if len(colored[cat]) < NUM_OBJECTS:
            colored[cat].append(objects[it].clone())
        it += 1

        done = True
        for i in range(4):
            if len(colored[i]) < NUM_OBJECTS:
                done = False
                break

for i in range(4):
    colored[i] = torch.stack(colored[i])
colored = torch.stack(colored)
colored = colored.reshape(-1, 42, 42)
colored = (colored - colored.min()) / (colored.max() - colored.min())
cm = plt.cm.plasma
colored = torch.tensor(cm(colored.numpy()), dtype=torch.float).permute(0, 3, 1, 2)[:, :3]
torchvision.utils.save_image(colored, "colored-objects.png", nrow=NUM_OBJECTS)
