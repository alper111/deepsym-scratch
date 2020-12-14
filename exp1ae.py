import os
import argparse

import torch
import yaml
from sklearn.tree import DecisionTreeClassifier

import data
import blocks

parser = argparse.ArgumentParser("Train effect prediction models.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
encoder = blocks.build_encoder(opts, 1)
decoder = blocks.MLP([opts["code1_dim"]] + [opts["hidden_dim"]] * opts["depth"] + [1764]).to("cuda")
encoder.to("cuda")
decoder.to("cuda")
optimizer = torch.optim.Adam(lr=0.0001, params=[{"params": encoder.parameters()}, {"params": decoder.parameters()}], amsgrad=True)

transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size1"], shuffle=True)
testloader = torch.utils.data.DataLoader(trainset, batch_size=2400, shuffle=False)

criterion = torch.nn.MSELoss()

for e in range(opts["epoch1"]):
    epoch_loss = 0.0
    for i, sample in enumerate(loader):
        x = sample["observation"].to(opts["device"])
        code = encoder(x)
        x_bar = decoder(code).reshape(-1, 1, 42, 42)
        loss = criterion(x_bar, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("Epoch: %d, loss: %.5f" % (e+1, epoch_loss / (i+1)))

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=2400, shuffle=False)
sample = iter(testloader).next()

encoder.eval()
decoder.eval()
with torch.no_grad():
    codes = encoder(sample["observation"].to(opts["device"])).cpu()

assigns = torch.load("data/effect1_labels.pt")
category = torch.cat([codes, sample["action"]], dim=1)
tree = DecisionTreeClassifier()
tree.fit(category, assigns)
preds = tree.predict(category)
print((torch.tensor(preds) == assigns).sum().float() / len(assigns), file=open(os.path.join(opts["save"], "results.txt"), "w"))
