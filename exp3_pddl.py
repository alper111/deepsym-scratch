"""
Translate decision tree rules to PDDL statements.
"""
import os
import argparse
import pickle
import utils


parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-s", help="save location", type=str, required=True)
parser.add_argument("-p", help="PPDDL (1) or PDDL (0). Default 1.", default=1, type=int)
args = parser.parse_args()

NUM_BITS = 14

save_name = os.path.join(args.s, "domain_mnist.pddl")
if os.path.exists(save_name):
    os.remove(save_name)

PROBABILISTIC = True if args.p == 1 else False

tree = pickle.load(open(os.path.join(args.s, "tree.pkl"), "rb"))
if args.p == 1:
    domain_name = "pdomain_mnist.pddl"
else:
    domain_name = "ddomain_mnist.pddl"
file_loc = os.path.join(args.s, domain_name)
if os.path.exists(file_loc):
    os.remove(file_loc)

action_features = list(range(NUM_BITS, NUM_BITS+4))
action_names = ["move_right", "move_up", "move_left", "move_down"]
actions = {}
for i in range(NUM_BITS+1, NUM_BITS+5):
    actions[i] = action_names.pop(0)
pddl_code = utils.tree_to_code_v2(tree, actions, PROBABILISTIC, NUM_BITS, 0)
pretext = "(define (domain mnist)\n"
pretext += "\t(:requirements :typing :negative-preconditions :conditional-effects :disjunctive-preconditions"
if PROBABILISTIC:
    pretext += " :probabilistic-effects"
pretext += ")"
pretext += "\n\t(:predicates"
for i in range(NUM_BITS):
    pretext += "\n\t\t(z%d)" % i
pretext += "\n\t)"
print(pretext, file=open(file_loc, "a"))

action_template = "\t(:action %s%d"
it = 0
for (precond, effect, action_name) in pddl_code:
    if len(precond) > 0:
        print(action_template % (action_name, it), file=open(file_loc, "a"))
        print("\t\t"+precond, file=open(file_loc, "a"))
        print("\t\t"+effect, file=open(file_loc, "a"))
        print("\t)", file=open(file_loc, "a"))
        it += 1
print(")", file=open(file_loc, "a"))
