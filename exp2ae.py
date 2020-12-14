import os
import argparse
import time

import torch
import yaml

import data
import blocks
import utils

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
        x = sample["observation"].to("cuda")
        code = encoder(x)
        x_bar = decoder(code).reshape(-1, 1, 42, 42)
        loss = criterion(x_bar, x)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    print("Epoch: %d, loss: %.5f" % (e+1, epoch_loss / (i+1)))

encoder.eval()
decoder.eval()

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=2400, shuffle=False)
sample = iter(testloader).next()
objects = sample["observation"].reshape(5, 10, 3, 4, 4, opts["size"], opts["size"])
objects = objects[:, :, 0].reshape(-1, 1, opts["size"], opts["size"])
with torch.no_grad():
    codes = encoder(objects.to(opts["device"]))
codes = codes.reshape(5, 160, opts["code1_dim"])
code_table = torch.zeros(5, 2**opts["code1_dim"])
for i in range(5):
    for j in range(160):
        code_table[i][utils.binary_to_decimal(codes[i, j])] += 1

torch.save(encoder.cpu().state_dict(), os.path.join(opts["save"], "encoder.pth"))
torch.save(decoder.cpu().state_dict(), os.path.join(opts["save"], "decoder.pth"))
print(code_table, file=open(os.path.join(opts["save"], "results.txt"), "w"))
