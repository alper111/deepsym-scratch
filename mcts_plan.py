import os
import argparse
import pickle

import yaml
import torch
import numpy as np

import mcts

parser = argparse.ArgumentParser("Make plan.")
parser.add_argument("-opts", help="option file", type=str, required=True)
parser.add_argument("-goal", help="goal state", type=str, default="(H3) (S4)")
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))

# scene information
codes1 = torch.load(os.path.join(opts["save"], "codes1.pth"))
codes2 = torch.load(os.path.join(opts["save"], "codes2.pth"))
objs = ["O{}".format(i+1) for i in range(len(codes1))]
# load the forward model
tree = pickle.load(open(os.path.join(opts["save"], "tree.pkl"), "rb"))
effects = list(np.load(os.path.join(opts["save"], "effect_names.npy")))
stack_idx = list(filter(lambda i: effects[i][0] == "s", range(len(effects))))
insert_idx = list(filter(lambda i: effects[i][0] == "i", range(len(effects))))
# parse goal
goal = args.goal.split(" ")
height = int(goal[0].strip("H()"))
stack = int(goal[1].strip("S()"))

f = mcts.ForwardDynamics(tree, codes1, codes2, stack_idx, insert_idx)
x = mcts.State(stack=[], inserts=[], picks=objs, drops=[], goal=(height, stack))
MCTS = mcts.MCTSNode(None, x, x.get_available_actions(), f)

MCTS.run(2000, 1)

state, plan, prob = MCTS.plan()
print("plan probability: %.2f" % prob)
plan_txt = ["stack %s %s" % (state.stack[i], state.stack[i+1]) for i in range(len(state.stack)-1)]
print("\n".join(plan_txt))
