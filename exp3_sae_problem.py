import os
import argparse
from copy import deepcopy

import torch
import torchvision
import matplotlib.pyplot as plt

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
encoder.load_state_dict(torch.load(os.path.join(args.s, "encoder.pt")))
encoder.eval()

for p in encoder.parameters():
    p.requires_grad = False

env = TilePuzzleMNIST(size=SIZE, permutation="replacement")

# env = TilePuzzleMNIST(size=SIZE, permutation=torch.tensor([2, 4, 1, 6, 7, 3, 8, 0, 5]))
# new_env = deepcopy(env)
x_init = env.state().unsqueeze(0)
torchvision.utils.save_image(x_init, "out/%dpuzzle_init.png" % (SIZE**2-1), nrow=1)


# new_env.step(1)
# new_env.step(0)
# new_env.step(0)
# new_env.step(0)
# new_env.step(1)
# x_goal = new_env.state().unsqueeze(0)

# generate goal encoding by randomly sampling goals
x_goal = env.random_goal_state().unsqueeze(0)

# x_goal = env.goal_state().unsqueeze(0)
# generate goal encoding by averaging goals
# r_idx = torch.randint(0, 9, ()).item()
# x_goal = torch.stack([env.avg_goal_state(r_idx) for i in range(1000)])

# torchvision.utils.save_image(x_goal, "out/%dpuzzle_goal.png" % (SIZE**2-1), nrow=1)
torchvision.utils.save_image(x_goal.mean(dim=0).unsqueeze(0), "out/%dpuzzle_goal.png" % (SIZE**2-1), nrow=1)

z_init = encoder(x_init).round().int()[0]
z_goal = encoder(x_goal).round().int()[0]
# z_goal = encoder(x_goal).mean(dim=0)
tau = 0.05
print(z_init)
print(["%.3f" % i for i in z_goal])

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
    if (z_i < (1-tau)) and (z_i > tau):
        print("This i skipped:", z_i)
        continue

    if z_i < 0.5:
        print(" (not (z%d))" % i, file=open(save_name, "a"), end="")
    else:
        print(" (z%d)" % i, file=open(save_name, "a"), end="")
print("))\n)", file=open(save_name, "a"))

fig, ax = plt.subplots(1, 2)
ax[0].set_title("Initial")
ax[0].imshow(x_init[0].permute(1, 2, 0), cmap="gray")
ax[1].set_title("Goal")
ax[1].imshow(x_goal.mean(dim=0).permute(1, 2, 0), cmap="gray")
plt.show()

planner = mGPT(rounds=1000, max_time=600, heuristic="ff")
domain_file = os.path.join(args.s, "pdomain_mnist.pddl")
problem_file = os.path.join(args.s, "problem_mnist.pddl")
valid, output = planner.find_plan(domain_file, problem_file)
if valid:
    print("Valid:", output.path)
    print(output.path, file=open("out/%dpuzzle_plan.txt" % (SIZE**2-1), "w"))
else:
    print("Invalid:", output)
