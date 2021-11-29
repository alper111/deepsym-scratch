import argparse
import os

import torch
import yaml
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import data
from models import DeepSymbolGenerator
from blocks import MLP, build_encoder

parser = argparse.ArgumentParser("test encoded model.")
parser.add_argument("-ckpt", help="checkpoint folder path.", type=str)
args = parser.parse_args()

file_loc = os.path.join(args.ckpt, "opts.yaml")
opts = yaml.safe_load(open(file_loc, "r"))
opts["device"] = "cpu"

encoder = build_encoder(opts, 2).to(opts["device"])
decoder = MLP([opts["code2_dim"]+opts["code1_dim"]*2+1] + [opts["hidden_dim"]]*opts["depth"] + [6]).to(opts["device"])
# we can omit the submodule since we ll only use the encoder
model = DeepSymbolGenerator(encoder, decoder, [], opts["device"], 0.001, os.path.join(opts["save"], "2"))
model.load("_best")
model.eval_mode()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)
trainset.train = False
loader = torch.utils.data.DataLoader(trainset, batch_size=36, shuffle=True)
objects = iter(loader).next()["state"]
with torch.no_grad():
    codes = model.encode(objects).round()

fig, ax = plt.subplots(6, 6, figsize=(10, 6))
for i in range(6):
    for j in range(6):
        idx = i * 6 + j
        ax[i, j].imshow(objects[idx].permute(1, 0, 2).reshape(objects.shape[3], objects.shape[3]*2)*0.0094+0.279)
        ax[i, j].axis("off")
        ax[i, j].set_title(codes[idx].numpy())
plt.show()
pp = PdfPages("paired.pdf")
pp.savefig(fig)
pp.close()
