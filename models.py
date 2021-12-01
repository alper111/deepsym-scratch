import os
import time

import torch

import utils
from blocks import MLP, build_encoder


class DeepSymbolGenerator:
    """DeepSym model from https://arxiv.org/abs/2012.02532"""

    def __init__(self, encoder: torch.nn.Module, decoder: torch.nn.Module,
                 subnetworks: list, device: str, lr: float,
                 path: str, coeff: float = 1.0):
        """
        Parameters
        ----------
        encoder : torch.nn.Module
            Encoder network.
        decoder : torch.nn.Module
            Decoder network.
        subnetworks : list of torch.nn.Module
            Optional list of subnetworks to use their output as input
            for the decoder.
        device : str
            The device of the networks.
        lr : float
            Learning rate.
        path : str
            Save and load path.
        coeff : float
            A hyperparameter to increase to speed of convergence when there
            are lots of zero values in the effect prediction (e.g. tile puzzle).
        """
        self.device = device
        self.coeff = coeff
        self.encoder = encoder
        self.decoder = decoder
        self.subnetworks = subnetworks

        self.optimizer = torch.optim.Adam(lr=lr, params=[
            {"params": self.encoder.parameters()},
            {"params": self.decoder.parameters()}])

        self.criterion = torch.nn.MSELoss()
        self.iteration = 0
        self.path = path

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given a state, return its encoding with the current
        encoder (i.e. no subnetwork code).

        Parameters
        ----------
        x : torch.Tensor
            The state tensor.

        Returns
        -------
        h : torch.Tensor
            The code of the given state.
        """
        h = self.encoder(x.to(self.device))
        return h

    def concat(self, sample: dict) -> torch.Tensor:
        """
        Given a sample, return the concatenation of the encoders'
        output and the action vector.

        Parameters
        ----------
        sample : dict
            The input dictionary. This dict should containt following
            keys: `state` and `action`.

        Returns
        -------
        z : torch.Tensor
            The concatenation of the encoder's output, subnetworks'
            encoders output, and the action vector (i.e. the input
            of the decoder).
        """
        h = []
        x = sample["state"]
        h.append(self.encode(x))
        for network in self.subnetworks:
            with torch.no_grad():
                h.append(network.encode(x))
        h.append(sample["action"].to(self.device))
        z = torch.cat(h, dim=-1)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Given a code, return the effect.

        Parameters
        ----------
        z : torch.Tensor
            The code tensor.

        Returns
        -------
        e : torch.Tensor
            The effect tensor.
        """
        e = self.decoder(z)
        return e

    def forward(self, sample):
        z = self.concat(sample)
        e = self.decode(z)
        return z, e

    def loss(self, sample):
        e_truth = sample["effect"].to(self.device)
        _, e_pred = self.forward(sample)
        L = self.criterion(e_pred, e_truth)*self.coeff
        return L

    def one_pass_optimize(self, loader):
        avg_loss = 0.0
        start = time.time()
        for i, sample in enumerate(loader):
            self.optimizer.zero_grad()
            L = self.loss(sample)
            L.backward()
            self.optimizer.step()
            avg_loss += L.item()
            self.iteration += 1
        end = time.time()
        avg_loss /= (i+1)
        time_elapsed = end-start
        return avg_loss, time_elapsed

    def train(self, epoch, loader):
        best_loss = 1e100
        for e in range(epoch):
            epoch_loss, time_elapsed = self.one_pass_optimize(loader)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save("_best")
            print(f"epoch={e+1}, iter={self.iteration}, loss={epoch_loss:.5f}, elapsed={time_elapsed:.2f}")
            self.save("_last")

    def load(self, ext):
        encoder_path = os.path.join(self.path, "encoder"+ext+".ckpt")
        decoder_path = os.path.join(self.path, "decoder"+ext+".ckpt")
        encoder_dict = torch.load(encoder_path)
        decoder_dict = torch.load(decoder_path)
        self.encoder.load_state_dict(encoder_dict)
        self.decoder.load_state_dict(decoder_dict)

    def save(self, ext):
        encoder_dict = self.encoder.eval().cpu().state_dict()
        decoder_dict = self.decoder.eval().cpu().state_dict()
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        encoder_path = os.path.join(self.path, "encoder"+ext+".ckpt")
        decoder_path = os.path.join(self.path, "decoder"+ext+".ckpt")
        torch.save(encoder_dict, encoder_path)
        torch.save(decoder_dict, decoder_path)
        self.encoder.train().to(self.device)
        self.decoder.train().to(self.device)

    def print_model(self, space=0, encoder_only=False):
        utils.print_module(self.encoder, "Encoder", space)
        if not encoder_only:
            utils.print_module(self.decoder, "Decoder", space)
        if len(self.subnetworks) != 0:
            print("-"*15)
            print("  Subnetworks  ")
            print("-"*15)

        tab_length = 4
        for i, network in enumerate(self.subnetworks):
            print(" "*tab_length+"%d:" % (i+1))
            network.print_model(space=space+tab_length, encoder_only=True)
            print()

    def eval_mode(self):
        self.encoder.eval()
        self.decoder.eval()

    def train_mode(self):
        self.encoder.train()
        self.decoder.train()


class EffectRegressorMLP:

    def __init__(self, opts):
        self.device = torch.device(opts["device"])

        if "discrete" not in opts:
            opts["discrete"] = True
        if "gumbel" not in opts:
            opts["gumbel"] = False

        self.encoder1 = build_encoder(opts, 1).to(self.device)
        self.encoder2 = build_encoder(opts, 2).to(self.device)
        self.decoder1 = MLP([opts["code1_dim"] + 3] + [opts["hidden_dim"]] * opts["depth"] + [3]).to(self.device)
        self.decoder2 = MLP([opts["code2_dim"] + opts["code1_dim"]*2] + [opts["hidden_dim"]] * opts["depth"] + [6]).to(self.device)
        self.optimizer1 = torch.optim.Adam(lr=opts["learning_rate1"],
                                           params=[
                                               {"params": self.encoder1.parameters()},
                                               {"params": self.decoder1.parameters()}],
                                           amsgrad=True)

        self.optimizer2 = torch.optim.Adam(lr=opts["learning_rate2"],
                                           params=[
                                               {"params": self.encoder2.parameters()},
                                               {"params": self.decoder2.parameters()}],
                                           amsgrad=True)

        self.criterion = torch.nn.MSELoss()
        self.iteration = 0
        self.save_path = opts["save"]

    def predict1(self, sample):
        obs = sample["state"].to(self.device)
        action = sample["action"].to(self.device)

        h = self.encoder1(obs)
        h_aug = torch.cat([h, action], dim=-1)
        effect_pred = self.decoder1(h_aug)
        return h, effect_pred

    def predict2(self, sample):
        obs = sample["state"].to(self.device)
        with torch.no_grad():
            h1 = self.encoder1(obs.reshape(-1, 1, obs.shape[2], obs.shape[3]))
        h1 = h1.reshape(obs.shape[0], -1)
        h2 = self.encoder2(obs)
        h_aug = torch.cat([h1, h2], dim=-1)
        effect_pred = self.decoder2(h_aug)
        return h_aug, effect_pred

    def loss1(self, sample):
        _, prediction = self.predict1(sample)
        return self.criterion(prediction, sample["effect"].to(self.device))

    def loss2(self, sample):
        _, prediction = self.predict2(sample)
        return self.criterion(prediction, sample["effect"].to(self.device))

    def one_pass_optimize(self, loader, level):
        running_avg_loss = 0.0
        for i, sample in enumerate(loader):
            if level == 1:
                self.optimizer1.zero_grad()
                loss = self.loss1(sample)
                loss.backward()
                self.optimizer1.step()
            else:
                self.optimizer2.zero_grad()
                loss = self.loss2(sample)
                loss.backward()
                self.optimizer2.step()
            running_avg_loss += loss.item()
            self.iteration += 1
        return running_avg_loss/i

    def train(self, epoch, loader, level):
        best_loss = 1e100
        for e in range(epoch):
            epoch_loss = self.one_pass_optimize(loader, level)
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                self.save(self.save_path, "_best", level)
            print("Epoch: %d, iter: %d, loss: %.4f" % (e+1, self.iteration, epoch_loss))
            self.save(self.save_path, "_last", level)

    def load(self, path, ext, level):
        if level == 1:
            encoder = self.encoder1
            decoder = self.decoder1
        else:
            encoder = self.encoder2
            decoder = self.decoder2

        encoder_dict = torch.load(os.path.join(path, "encoder"+str(level)+ext+".ckpt"))
        decoder_dict = torch.load(os.path.join(path, "decoder"+str(level)+ext+".ckpt"))
        encoder.load_state_dict(encoder_dict)
        decoder.load_state_dict(decoder_dict)

    def save(self, path, ext, level):
        if level == 1:
            encoder = self.encoder1
            decoder = self.decoder1
        else:
            encoder = self.encoder2
            decoder = self.decoder2

        encoder_dict = encoder.eval().cpu().state_dict()
        decoder_dict = decoder.eval().cpu().state_dict()
        torch.save(encoder_dict, os.path.join(path, "encoder"+str(level)+ext+".ckpt"))
        torch.save(decoder_dict, os.path.join(path, "decoder"+str(level)+ext+".ckpt"))
        encoder.train().to(self.device)
        decoder.train().to(self.device)

    def print_model(self, level):
        encoder = self.encoder1 if level == 1 else self.encoder2
        decoder = self.decoder1 if level == 1 else self.decoder2
        print("="*10+"ENCODER"+"="*10)
        print(encoder)
        print("parameter count: %d" % utils.get_parameter_count(encoder))
        print("="*27)
        print("="*10+"DECODER"+"="*10)
        print(decoder)
        print("parameter count: %d" % utils.get_parameter_count(decoder))
        print("="*27)

    def eval_mode(self):
        self.encoder1.eval()
        self.encoder2.eval()
        self.decoder1.eval()
        self.decoder2.eval()

    def train_mode(self):
        self.encoder1.train()
        self.encoder2.train()
        self.decoder1.train()
        self.decoder2.train()
