import os
import yaml

sizes = [1, 2, 3, 4, 5]
N = 5
for i in range(N):
    for s in sizes:
        file = open(os.path.join("opts_trainjob.yml"), "w")
        opts = {}
        opts["batch_norm"] = True
        opts["cnn"] = True

        opts["epoch1"] = 300
        opts["batch_size1"] = 128
        opts["learning_rate1"] = 0.00005
        opts["code1_dim"] = s
        opts["filters1"] = [1, 32, 64, 128, 256]

        opts["epoch2"] = 300
        opts["batch_size2"] = 128
        opts["learning_rate2"] = 0.00005
        opts["code2_dim"] = 1
        opts["filters2"] = [2, 32, 64, 128, 256]

        opts["hidden_dim"] = 128
        opts["depth"] = 2
        opts["size"] = 42
        opts["load"] = None
        opts["save"] = "save/single_unit%d_%d" % (s, i)
        opts["device"] = "cuda"

        opts["discrete"] = True
        opts["gumbel"] = True
        opts["temperature"] = 1.0
        opts["gumbel_hard"] = False

        yaml.dump(opts, file)
        file.close()
        print("Started training with %d units, #%d" % (s, i))
        os.system("python train.py -opts opts_trainjob.yml")
