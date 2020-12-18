import argparse
import rospy
import numpy as np
import torch

from simtools.rosutils import RosNode


parser = argparse.ArgumentParser("Generate a scene.")
parser.add_argument("-uri", help="master uri", type=str, default="http://localhost:11311")
args = parser.parse_args()

node = RosNode("recognize_scene", args.uri)
node.stopSimulation()
rospy.sleep(1.0)
node.startSimulation()
rospy.sleep(1.0)

np.random.seed(13337)
# GENERATE A RANDOM SCENE
NUM_OBJECTS = 5
objTypes = np.random.randint(1, 6, (NUM_OBJECTS, ))
objSizes = np.random.uniform(1.0, 2, (5, )).tolist()
locations = np.array([
    [-0.69, -0.09],
    [-0.9, -0.35],
    [-0.45, 0.175],
    [-0.45, -0.35],
    [-0.9, 0.175]
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
