import os
import argparse
import pickle
import utils


parser = argparse.ArgumentParser("learn pddl rules from decision tree.")
parser.add_argument("-t", help="tree file", type=str, required=True)
parser.add_argument("-p", help="PPDDL (1) or PDDL (0). Default 1.", default=1, type=int)
parser.add_argument("-s", help="save location", type=str, required=True)
args = parser.parse_args()

save_name = os.path.join(args.s, "domain_mnist.pddl")
if os.path.exists(save_name):
    os.remove(save_name)

PROBABILISTIC = True if args.p == 1 else False

# TODO: make this generic
tree = pickle.load(open(args.t, "rb"))
if args.p == 1:
    domain_name = "pdomain_mnist.pddl"
else:
    domain_name = "ddomain_mnist.pddl"
file_loc = os.path.join(args.s, domain_name)
if os.path.exists(file_loc):
    os.remove(file_loc)

# input_features = list(range(21))
action_features = list(range(13, 17))
action_names = ["move_right", "move_up", "move_left", "move_down"]
pddl_code = utils.tree_to_code_v2(tree, action_features, PROBABILISTIC)
pretext = "(define (domain mnist)\n"
pretext += "\t(:requirements :typing :negative-preconditions :conditional-effects :disjunctive-preconditions"
if PROBABILISTIC:
    pretext += " :probabilistic-effects"
pretext += ")"
pretext += "\n\t(:predicates"

for i in range(17):
    pretext += "\n\t\t(z%d)" % i
pretext += "\n\t)"
print(pretext, file=open(file_loc, "a"))

action_template = "\t(:action aux%d"
for i, (precond, effect) in enumerate(pddl_code):
    print(action_template % i, file=open(file_loc, "a"))
    print("\t\t"+precond, file=open(file_loc, "a"))
    print("\t\t"+effect, file=open(file_loc, "a"))
    print("\t)", file=open(file_loc, "a"))
for i, a_i in enumerate(action_features):
    print("\t(:action %s" % action_names[i], file=open(file_loc, "a"))
    print("\t\t:precondition (and", end="", file=open(file_loc, "a"))
    for j in action_features:
        print(" (not (z%d))" % j, end="", file=open(file_loc, "a"))
    print(")", file=open(file_loc, "a"))

    print("\t\t:effect (and", end="", file=open(file_loc, "a"))
    for j in action_features:
        if a_i == j:
            print(" (z%d)" % j, end="", file=open(file_loc, "a"))
        else:
            print(" (not (z%d))" % j, end="", file=open(file_loc, "a"))
    print(")", file=open(file_loc, "a"))
    print("\t)", file=open(file_loc, "a"))
print(")", file=open(file_loc, "a"))
