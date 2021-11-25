import pickle

import torch
from sklearn.tree import DecisionTreeClassifier

from models import DeepSymbolGenerator
from data import TilePuzzleData
import blocks
import utils

STATE = torch.load("data/tile_state.pt") / 255.0
EFFECT = torch.load("data/tile_effect.pt") / 255.0
ACTION = torch.load("data/tile_action.pt")

BN = True
NUM_ACTIONS = 4
NUM_BITS = 13
BATCH_SIZE = 1000

encoder = torch.nn.Sequential(
    blocks.ConvBlock(in_channels=1, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=64, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=128, out_channels=256, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvBlock(in_channels=256, out_channels=512, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.Avg([2, 3]),
    blocks.MLP([512, NUM_BITS]),
    blocks.GumbelSigmoidLayer(hard=False, T=1.0)
)

decoder = torch.nn.Sequential(
    blocks.MLP([NUM_BITS+NUM_ACTIONS, 512]),
    blocks.Reshape([-1, 512, 1, 1]),
    blocks.ConvTransposeBlock(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=0, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0, batch_norm=BN),
    torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)
)
encoder.to("cuda")
decoder.to("cuda")

model = DeepSymbolGenerator(encoder=encoder, decoder=decoder, subnetworks=[],
                            device="cuda", lr=1e-4, path="save/tile_puzzle", coeff=9.0)

model.load("_best")
model.eval_mode()
for p in model.encoder.parameters():
    p.requires_grad = False
for p in model.decoder.parameters():
    p.requires_grad = False

data = TilePuzzleData("./data")
loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE)

train_X = []
train_Y = []
for e in range(1):
    print(f"Epoch {e+1}")
    for i, sample in enumerate(loader):
        sample["state"] = sample["state"].to("cuda")
        sample["action"] = sample["action"].to("cuda")

        z = model.concat(sample).round()
        e = model.decode(z)
        state_next = (sample["state"]+e).clamp(0., 1.)
        z_next = model.encode(state_next)
        train_X.append(z.round().cpu())
        train_Y.append(z_next.round().cpu())

train_X = torch.cat(train_X, dim=0)
train_Y = torch.cat(train_Y, dim=0)
print("X shape:", train_X.shape)
print("Y shape:", train_Y.shape)
train_Y = utils.binary_to_decimal_tensor(train_Y)

tree = DecisionTreeClassifier()
tree.fit(train_X, train_Y)

# preds = []
# for i in range(100):
#     preds.append(torch.tensor(tree.predict(train_X[i*1000:(i+1)*1000])))
# preds = torch.cat(preds, dim=0)

preds = tree.predict(train_X)

print(tree.get_depth())
print(tree.get_n_leaves())

accuracy = (torch.tensor(preds) == train_Y).sum().float() / len(train_Y) * 100
print("%.1f" % accuracy)

file = open("temp_tree.pkl", "wb")
pickle.dump(tree, file)
file.close()
