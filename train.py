import os
import argparse
import time

import yaml
import torch

from models import DeepSymbolGenerator
from blocks import MLP, build_encoder, ChannelWrapper
import data

parser = argparse.ArgumentParser("Train effect prediction models.")
parser.add_argument("-opts", help="option file", type=str, required=True)
args = parser.parse_args()

opts = yaml.safe_load(open(args.opts, "r"))
if not os.path.exists(opts["save"]):
    os.makedirs(opts["save"])
opts["time"] = time.asctime(time.localtime(time.time()))
file = open(os.path.join(opts["save"], "opts.yaml"), "w")
yaml.dump(opts, file)
file.close()
print(yaml.dump(opts))

device = torch.device(opts["device"])

# load the single object interaction data
transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.SingleObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size1"], shuffle=True)

# train for single object interaction data
encoder1 = build_encoder(opts, 1).to(opts["device"])
decoder1 = MLP([opts["code1_dim"]+3] + [opts["hidden_dim"]]*opts["depth"] + [3]).to(opts["device"])
model1 = DeepSymbolGenerator(encoder1, decoder1, [], opts["device"], opts["learning_rate1"], os.path.join(opts["save"], "1"))
model1.print_model()
model1.train(opts["epoch1"], loader)
# load the best encoder1
model1.load("_best")

# load the paired object interaction data
transform = data.default_transform(size=opts["size"], affine=True, mean=0.279, std=0.0094)
trainset = data.PairedObjectData(transform=transform)
loader = torch.utils.data.DataLoader(trainset, batch_size=opts["batch_size2"], shuffle=True)

model1.encoder = ChannelWrapper(model1.encoder)
model1.eval_mode()

# train for paired object interaction_data
encoder2 = build_encoder(opts, 2).to(opts["device"])
decoder2 = MLP([opts["code2_dim"]+opts["code1_dim"]*2+1] + [opts["hidden_dim"]]*opts["depth"] + [6]).to(opts["device"])
model2 = DeepSymbolGenerator(encoder2, decoder2, [model1], opts["device"], opts["learning_rate1"], os.path.join(opts["save"], "2"))
model2.print_model()
model2.train(opts["epoch2"], loader)
