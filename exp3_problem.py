import os
import argparse

import torch
import matplotlib.pyplot as plt

from models import DeepSymbolGenerator
from envs import TilePuzzleMNIST
import blocks
from mgpt_planner import mGPT

parser = argparse.ArgumentParser("convert to problem.")
parser.add_argument("-s", help="save location", type=str, required=True)
args = parser.parse_args()

save_name = os.path.join(args.s, "problem_mnist.pddl")
if os.path.exists(save_name):
    os.remove(save_name)

BN = True
NUM_ACTIONS = 4
NUM_BITS = 13
SIZE = 3

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
model = DeepSymbolGenerator(encoder=encoder, decoder=decoder, subnetworks=[],
                            device="cpu", lr=1e-4, path="save/tile_puzzle", coeff=9.0)


model.load("_best")
model.eval_mode()
for p in model.encoder.parameters():
    p.requires_grad = False
for p in model.decoder.parameters():
    p.requires_grad = False

env = TilePuzzleMNIST()
x_init = env.state().unsqueeze(0)
x_goal = env.random_goal_state().unsqueeze(0)
z_init = model.encode(x_init).round().int()[0]
z_goal = model.encode(x_goal).round().int()[0]
print(z_init)
print(z_goal)

print("(define (problem tilepuzzle) (:domain mnist)", file=open(save_name, "a"))
print("\t(:init ", file=open(save_name, "a"), end="")

for i, z_i in enumerate(z_init):
    if z_i == 0:
        continue
    else:
        print(" (z%d)" % i, file=open(save_name, "a"), end="")
print(")", file=open(save_name, "a"))
print("\t(:goal (and", file=open(save_name, "a"), end="")
for i, z_i in enumerate(z_goal):
    if z_i == 0:
        print(" (not (z%d))" % i, file=open(save_name, "a"), end="")
    else:
        print(" (z%d)" % i, file=open(save_name, "a"), end="")
print("))\n)", file=open(save_name, "a"))


planner = mGPT(rounds=500)
valid, output = planner.find_plan("save/tile_puzzle/pdomain_mnist.pddl", "save/tile_puzzle/problem_mnist.pddl")
print(output.path)

fig, ax = plt.subplots(1, 2)
ax[0].set_title("Initial")
ax[0].imshow(x_init[0].permute(1, 2, 0), cmap="gray")
ax[1].set_title("Goal")
ax[1].imshow(x_goal[0].permute(1, 2, 0), cmap="gray")
plt.show()
