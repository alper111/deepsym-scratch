import sys

import torch
import yaml
import numpy as np
from sklearn.tree import DecisionTreeClassifier

import data
import blocks
import utils

opts = yaml.safe_load(open(sys.argv[1], "r"))
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
    print("loss: %.5f" % (epoch_loss / (i+1)))

encoder.eval()
sample = iter(testloader).next()
x = sample["observation"].to("cuda")
x = x.reshape(-1, 1, 42, 42)
with torch.no_grad():
    code = encoder(x)

category = torch.cat([code.cpu(), sample["action"]], dim=1)
assigns = torch.load("newlabels.pt")

print(category.shape)
tree = DecisionTreeClassifier()
tree.fit(category, assigns)
print(tree.get_depth())
preds = tree.predict(category)
print((torch.tensor(preds) == assigns).sum().float() / len(assigns))
exit()

codes = []
for c in code:
    codes.append(utils.binary_to_decimal(c))
codes = np.array(codes).reshape(5, 10)
print(codes)
