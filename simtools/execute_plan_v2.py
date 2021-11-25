import argparse

import comm

parser = argparse.ArgumentParser("Parse plan.")
parser.add_argument("-p", help="plan file", type=str, required=True)
parser.add_argument("-level", help="same level or not.", type=int, required=True)
args = parser.parse_args()

env = comm.Communication(20000)
env.open_connection()
env._initialize_handles()

file = open(args.p, "r")
lines = file.readlines()
N = int(lines[0])
objNames = []
objLocs = []
objSizes = []
for i in range(N):
    name, x, y, size = lines[i+1].split()
    objNames.append(name)
    objLocs.append([float(x), float(y)])
    objSizes.append(float(size))

print("Plan success probability: %.2f" % float(lines[N+1].split(":")[1]))
if lines[N+1] == "not found.":
    print("Cannot find a plan which satisfies the objective.")
    exit()

base_level = 0.7
WAIT = 20
if args.level:
    base_level += objSizes[objNames.index(lines[N+2].split()[1])]
for p in lines[N+2:]:
    env.hand_open_pose(wait=WAIT)
    env.init_arm_pose(wait=WAIT)
    _, base, target = p.split()
    base_idx = objNames.index(base)
    target_idx = objNames.index(target)
    base_loc = objLocs[base_idx]
    target_loc = objLocs[target_idx]
    env.set_tip_pose(target_loc+[1.0], wait=WAIT)
    env.set_tip_pose(target_loc+[0.7+0.85*objSizes[target_idx]], wait=WAIT)
    env.hand_grasp_pose(wait=WAIT)
    if args.level:
        env.set_tip_pose(target_loc+[base_level+objSizes[target_idx]+0.05], wait=WAIT)
        env.set_tip_pose(base_loc+[base_level+objSizes[target_idx]+0.05], wait=4*WAIT)
    else:
        base_level += objSizes[base_idx]
        env.set_tip_pose(target_loc+[base_level+objSizes[target_idx]+0.05], wait=WAIT)
        env.set_tip_pose(base_loc+[base_level+objSizes[target_idx]+0.05], wait=4*WAIT)

    for i in range(10):
        env.step()
    env.hand_open_pose(wait=WAIT)
    objLocs[target_idx] = objLocs[base_idx]
env.init_arm_pose(wait=WAIT)
