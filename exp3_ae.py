"""Train SAE on MNIST 8-tile puzzle."""
import argparse
import os

import torch

from data import TilePuzzleData
import blocks
import utils

parser = argparse.ArgumentParser("Train SAE on Tile MNIST Env.")
parser.add_argument("-s", help="save folder", type=str, required=True)
args = parser.parse_args()

if not os.path.exists(args.s):
    os.makedirs(args.s)

BN = True
NUM_BITS = 13
NUM_EPOCH = 100
LR = 0.0001
BATCH_SIZE = 128
N = 100000
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

# 13-bit or 14-bit, 3x3
decoder = torch.nn.Sequential(
    blocks.MLP([NUM_BITS, 512]),
    blocks.Reshape([-1, 512, 1, 1]),
    blocks.ConvTransposeBlock(in_channels=512, out_channels=256, kernel_size=5, stride=1, padding=0, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=256, out_channels=128, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=128, out_channels=64, kernel_size=4, stride=2, padding=1, batch_norm=BN),
    blocks.ConvTransposeBlock(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=0, batch_norm=BN),
    torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=4, stride=2, padding=1)
)

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
decoder.to("cuda")
optimizer = torch.optim.Adam(lr=0.0001, params=[{"params": encoder.parameters()}, {"params": decoder.parameters()}], amsgrad=True)

data = TilePuzzleData("./data")
loader = torch.utils.data.DataLoader(data, batch_size=BATCH_SIZE, num_workers=12)

criterion = torch.nn.MSELoss()
prior = torch.tensor(0.1)
ALPHA = 1.0

print(encoder)
print(decoder)
for e in range(NUM_EPOCH):
    epoch_loss = 0.0
    for i, sample in enumerate(loader):
        x_i = sample["state"].to("cuda")
        z_i = encoder(x_i)
        x_bar_i = decoder(z_i)
        mse_loss = criterion(x_bar_i, x_i)
        kl_loss = utils.kl_bernoulli(z_i, prior).mean()
        loss = mse_loss + ALPHA * kl_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += mse_loss.item()
    print("Epoch: %d, loss: %.5f" % (e+1, epoch_loss / (i+1)))

torch.save(encoder.eval().cpu().state_dict(), os.path.join(args.s, "encoder.pt"))
torch.save(decoder.eval().cpu().state_dict(), os.path.join(args.s, "decoder.pt"))