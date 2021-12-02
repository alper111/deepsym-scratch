"""Train DeepSym on MNIST 8-tile puzzle."""
import argparse
import os

import torch

from models import DeepSymbolGenerator
from data import TilePuzzleData
import blocks

parser = argparse.ArgumentParser("Train DeepSym on Tile MNIST Env.")
parser.add_argument("-s", help="save folder", type=str, required=True)
args = parser.parse_args()

if not os.path.exists(args.s):
    os.makedirs(args.s)

BN = True
NUM_ACTIONS = 4
NUM_BITS = 14
NUM_EPOCH = 100
LR = 0.0001
BATCH_SIZE = 128
N = 500000
LOOP_PER_EPOCH = N // BATCH_SIZE

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
                            device="cuda", lr=1e-4, path=args.s, coeff=9.0)
model.print_model()

data = TilePuzzleData("./data")
loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE)
model.train(NUM_EPOCH, loader)
