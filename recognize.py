import os
import argparse
import yaml
import torch
from models import EffectRegressorMLP
import data
import utils


parser = argparse.ArgumentParser("Make plan.")
parser.add_argument("-opts", help="option file", type=str, required=True)
parser.add_argument("-goal", help="goal state", type=str, default="(H3) (S0)")
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
device = torch.device(opts["device"])

model = EffectRegressorMLP(opts)
model.load(opts["save"], "_best", 1)
model.load(opts["save"], "_best", 2)
model.encoder1.eval()
model.encoder2.eval()
# Homogeneous transformation matrix
H = torch.load("H.pt")
# object locations and sizes
locations = torch.load("location.pth")
objSizes = torch.load("sizes.pth")
# depth image
x = torch.load("depthimg.pth")
# estimated objects and locations
objs, locs, _ = utils.find_objects(x, opts["size"])
n_obj = len(objs)

transform = data.default_transform(size=opts["size"], affine=False, mean=0.279, std=0.0094)
for i, o in enumerate(objs):
    objs[i] = transform(o)[0]
objs = objs.to(device)

# homogeneous transformation of locations
locs = torch.cat([locs.float(), torch.ones(locs.shape[0], 1, device=locs.device)], dim=1)
locs = torch.matmul(locs, H.T)
locs = locs / locs[:, 2].reshape(-1, 1)

_, indices = torch.cdist(locs[:, :2], locations).min(dim=1)
obj_infos = []
comparisons = []
codes1 = torch.zeros(n_obj, opts["code1_dim"])
codes2 = torch.zeros(n_obj, n_obj)

# recognize objects
with torch.no_grad():
    for i, obj in enumerate(objs):
        cat = model.encoder1(obj.unsqueeze(0).unsqueeze(0))
        if "discrete" in opts and not opts["discrete"]:
            c1 = torch.load(os.path.join(opts["save"], "centroids_1.pth")).to(opts["device"])
            _, idx = torch.cdist(cat, c1).topk(k=2, largest=False)
            cat = torch.tensor(utils.decimal_to_binary(idx[0, 1], length=opts["code1_dim"])).unsqueeze(0)
        codes1[i] = cat
        print("Category: (%d %d), Location: (%.5f %.5f)" % (cat[0, 0], cat[0, 1], locations[indices[i], 0], locations[indices[i], 1]))
        info = {}
        info["name"] = "O{}".format(i+1)
        info["loc"] = (locations[indices[i], 0].item(), locations[indices[i], 1].item())
        info["size"] = objSizes[indices[i]]*0.1
        info["type"] = "objtype{}".format(utils.binary_to_decimal([int(cat[0, 0]), int(cat[0, 1])]))

        obj_infos.append(info)
        for j in range(n_obj):
            rel = model.encoder2(torch.stack([obj, objs[j]]).unsqueeze(0))
            if "discrete" in opts and not opts["discrete"]:
                c2 = torch.load(os.path.join(opts["save"], "centroids_2.pth")).to(opts["device"])
                _, idx = torch.cdist(rel, c2).topk(k=2, largest=False)
                rel = utils.decimal_to_binary(idx[0, 0], length=opts["code2_dim"])[0]
            codes2[i, j] = rel
            if i != j:
                if rel == -1:
                    comparisons.append("(relation0 O%d O%d)" % (i+1, j+1))
                else:
                    comparisons.append("(relation1 O%d O%d)" % (i+1, j+1))
print(obj_infos)
print(comparisons)

torch.save(codes1, os.path.join(opts["save"], "codes1.pth"))
torch.save(codes2, os.path.join(opts["save"], "codes2.pth"))

# print object information
file_obj = os.path.join(opts["save"], "objects.txt")
if os.path.exists(file_obj):
    os.remove(file_obj)
print(str(len(obj_infos)), file=open(file_obj, "a"))

# print problem definition
file_loc = os.path.join(opts["save"], "problem.pddl")
if os.path.exists(file_loc):
    os.remove(file_loc)
print("(define (problem dom1) (:domain stack)", file=open(file_loc, "a"))
object_str = "\t(:objects"
init_str = "\t(:init\n"
for obj_i in obj_infos:
    print("%s %.5f %.5f %.5f" % (obj_i["name"], obj_i["loc"][0], obj_i["loc"][1], obj_i["size"]), file=open(file_obj, "a"))
    object_str += " " + obj_i["name"]
    init_str += "\t\t(pickloc " + obj_i["name"] + ") (" + obj_i["type"] + " " + obj_i["name"] + ")\n"
object_str += ")"
for c_i in comparisons:
    init_str += "\t\t" + c_i + "\n"
init_str += "\t\t(H0)\n"
init_str += "\t\t(S0)\n"
init_str += "\t)"

goal_str = "\t(:goal (and %s (not (stacked)) (not (inserted))))\n)" % args.goal
print(object_str, file=open(file_loc, "a"))
print(init_str, file=open(file_loc, "a"))
print(goal_str, file=open(file_loc, "a"))
