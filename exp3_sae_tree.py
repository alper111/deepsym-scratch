"""
Learn a decision tree from SAE symbols.
"""
import pickle
import argparse
import os

import torch
from sklearn.tree import DecisionTreeClassifier

from data import TilePuzzleData
import blocks
import utils

parser = argparse.ArgumentParser("Train decision tree with the decoder input output.")
parser.add_argument("-s", help="model folder", type=str, required=True)
args = parser.parse_args()

BN = True
NUM_BITS = 13
BATCH_SIZE = 500
N = 100000

encoder = torch.nn.Sequential(
    blocks.ConvBlock(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.Avg([2, 3]),
    blocks.MLP([512, NUM_BITS]),
    blocks.GumbelSigmoidLayer(hard=False, T=1.0)
)

# 13-bit or 14-bit, 3x3
# decoder = torch.nn.Sequential(
#     blocks.MLP([NUM_BITS, 512]),
#     blocks.Reshape([-1, 512, 1, 1]),
#     blocks.ConvTransposeBlock(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=0, batch_norm=BN),
#     blocks.ConvTransposeBlock(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
#     blocks.ConvTransposeBlock(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
#     blocks.ConvTransposeBlock(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0, batch_norm=BN),
#     torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)
# )

# 15-bit, 4x4
# decoder = torch.nn.Sequential(
#     blocks.MLP([NUM_BITS, 512]),
#     blocks.Reshape([-1, 512, 1, 1]),
#     blocks.ConvTransposeBlock(in_channels=512, out_channels=256, kernel_size=7, stride=1, padding=0, batch_norm=BN),
#     blocks.ConvTransposeBlock(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
#     blocks.ConvTransposeBlock(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
#     blocks.ConvTransposeBlock(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1, batch_norm=BN),
#     torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)
# )

# 16-bit, 5x5
# decoder = torch.nn.Sequential(
#     blocks.MLP([NUM_BITS, 512]),
#     blocks.Reshape([-1, 512, 1, 1]),
#     blocks.ConvTransposeBlock(in_channels=512, out_channels=256, kernel_size=8, stride=1, padding=0, batch_norm=BN),
#     blocks.ConvTransposeBlock(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
#     blocks.ConvTransposeBlock(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=0, batch_norm=BN),
#     blocks.ConvTransposeBlock(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0, batch_norm=BN),
#     torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)
# )
encoder.to("cuda")
# decoder.to("cuda")

encoder.load_state_dict(torch.load(os.path.join(args.s, "encoder.pt")))
encoder.eval()
# decoder.eval()

for p in encoder.parameters():
    p.requires_grad = False
# for p in decoder.parameters():
    # p.requires_grad = False

data = TilePuzzleData("./data")
loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE)

train_X = []
train_Y = []
for e in range(1):
    print(f"Epoch {e+1}")
    for i, sample in enumerate(loader):
        sample["state"] = sample["state"].to("cuda")
        sample["effect"] = sample["effect"].to("cuda")
        sn = sample["state"] + sample["effect"]
        z = encoder(sample["state"]).round()
        zn = encoder(sn).round()
        state_next = (sample["state"]+e).clamp(0., 1.)
        # z_next = model.encode(state_next)
        train_X.append(z.cpu())
        train_Y.append(zn.cpu())

train_X = torch.cat(train_X, dim=0)
train_Y = torch.cat(train_Y, dim=0)
print("X shape:", train_X.shape)
print("Y shape:", train_Y.shape)
train_Y = utils.binary_to_decimal_tensor(train_Y)

tree = DecisionTreeClassifier()
tree.fit(train_X, train_Y)

preds = []
K = N // BATCH_SIZE
for i in range(K):
    preds.append(torch.tensor(tree.predict(train_X[i*BATCH_SIZE:(i+1)*BATCH_SIZE])))
preds = torch.cat(preds, dim=0)

# preds = tree.predict(train_X)

print(tree.get_depth())
print(tree.get_n_leaves())

accuracy = (torch.tensor(preds) == train_Y).sum().float() / len(train_Y) * 100
print("%.1f" % accuracy)

file = open(os.path.join(args.s, "tree.pkl"), "wb")
pickle.dump(tree, file)
file.close()
