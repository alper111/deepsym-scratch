import argparse
import rospy
import numpy as np
import torch

from simtools.rosutils import RosNode


parser = argparse.ArgumentParser("Generate a scene.")
parser.add_argument("-uri", help="master uri", type=str, default="http://localhost:11311")
parser.add_argument("-s", help="setting. 1, 2, 3, or 4.", type=int, required=True)
args = parser.parse_args()

node = RosNode("recognize_scene", args.uri)
node.stopSimulation()
rospy.sleep(1.0)
node.startSimulation()
rospy.sleep(1.0)

np.random.seed(13337)
# GENERATE A RANDOM SCENE
NUM_OBJECTS = 5
STABLE_OBJECTS = [2, 3, 5]
CUPS = [5, 6, 7, 8]  # same cup but with different colors

# Ensure that the goal is attainable by generating
# objects which can satify the goal. Otherwise, the
# planner outputs "cannot found plan" as expected.
# This is not a hack.

if args.s == 1:
    # SETTING 1 - (H4) (S4)
    # 4 stable objects, 1 random object
    objTypes = np.random.choice(STABLE_OBJECTS, 4).tolist() + [np.random.randint(1, 6)]
    objSizes = np.random.uniform(1.0, 2, (5, )).tolist()
elif args.s == 2:
    # SETTING 2 - (H3) (S4)
    # 1 huge cup, 3 stable objects, 1 random object
    objTypes = np.random.choice(CUPS, 1).tolist() + np.random.choice(STABLE_OBJECTS, 3).tolist() + [np.random.randint(1, 6)]
    objSizes = [2.0] + np.random.uniform(1.0, 1.7, (3, )).tolist() + [np.random.uniform(1., 2.)]
elif args.s == 3:
    # SETTING 3 - (H2) (S4)
    # 1 big cup, 1 medium cup, 2 small stable objects, 1 random object
    objTypes = np.random.choice(CUPS, 2, replace=False).tolist() + np.random.choice(STABLE_OBJECTS, 2).tolist() + [np.random.randint(1, 6)]
    objSizes = [2.1, 1.5] + np.random.uniform(1.0, 1.1, (2, )).tolist() + np.random.uniform(1., 2., (2,)).tolist()
elif args.s == 4:
    # SETTING 4 - (H1) (S4)
    # 1 big cup, 1 medium cup, 1 small cup, 1 small random object, 1 random object
    objTypes = np.random.choice(CUPS, 3, replace=False).tolist() + np.random.randint(1, 6, (2,)).tolist()
    objSizes = [2.1, 1.5, 1.08, 0.72] + [np.random.uniform(1., 2.0)]


locations = np.array([
    [-0.69, -0.09],
    [-0.79, -0.35],
    [-0.45, 0.175],
    [-0.45, -0.35],
    [-0.79, 0.175]
])
locations = locations[np.random.permutation(5)]
locations = locations[:NUM_OBJECTS].tolist()

for i in range(NUM_OBJECTS):
    node.generateObject(objTypes[i], objSizes[i], locations[i]+[objSizes[i]*0.05+0.7])
rospy.sleep(1.0)
locations = torch.tensor(locations, dtype=torch.float)
x = torch.tensor(node.getDepthImage(8), dtype=torch.float)
torch.save(objSizes, "sizes.pth")
torch.save(locations, "location.pth")
torch.save(x, "depthimg.pth")
