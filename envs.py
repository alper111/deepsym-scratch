import torch

MNIST_DATA = torch.load("data/mnist_data.pt")
MNIST_LABELS = torch.load("data/mnist_label.pt")
# EMNIST_DATA = torch.load("data/emnist_data.pt")
# EMNIST_LABELS = torch.load("data/emnist_label.pt")


class TilePuzzleMNIST:
    def __init__(self, permutation=None, size=3, dataset="mnist", random=False):
        self.action_names = [
            "move_right",
            "move_up",
            "move_left",
            "move_down"
        ]
        self.index = None
        self.location = None
        self.size = size
        self.random = random
        if permutation is not None:
            self.permutation = permutation
        elif size > 3:
            self.permutation = "replacement"
        else:
            self.permutation = None
        self.num_tile = size ** 2
        if dataset == "mnist":
            self.num_class = 10
        elif dataset == "emnist":
            self.num_class = 47

        self.reset(permutation=self.permutation)

    def step(self, action):
        row, col = self.location
        blank_idx = self.index[row, col].item()
        if action == 0:
            if self.location[1] != self.size-1:
                piece_idx = self.index[row, col+1].item()
                self.index[row, col] = piece_idx
                self.index[row, col+1] = blank_idx
                self.location[1] = col + 1

                self.permutation[row, col] = self.permutation[row, col+1]
                self.permutation[row, col+1] = 0
        elif action == 1:
            if self.location[0] != 0:
                piece_idx = self.index[row-1, col].item()
                self.index[row, col] = piece_idx
                self.index[row-1, col] = blank_idx
                self.location[0] = row - 1

                self.permutation[row, col] = self.permutation[row-1, col]
                self.permutation[row-1, col] = 0
        elif action == 2:
            if self.location[1] != 0:
                piece_idx = self.index[row, col-1].item()
                self.index[row, col] = piece_idx
                self.index[row, col-1] = blank_idx
                self.location[1] = col - 1

                self.permutation[row, col] = self.permutation[row, col-1]
                self.permutation[row, col-1] = 0
        elif action == 3:
            if self.location[0] != self.size-1:
                piece_idx = self.index[row+1, col].item()
                self.index[row, col] = piece_idx
                self.index[row+1, col] = blank_idx
                self.location[0] = row + 1

                self.permutation[row, col] = self.permutation[row+1, col]
                self.permutation[row+1, col] = 0
        return self.state()

    def reset(self, permutation=None):
        if permutation is None:
            # this only works in 3x3
            perm = torch.randperm(self.num_tile)
        elif permutation == "replacement":
            perm = self._draw_with_replacement()
        else:
            perm = permutation
        self.index = torch.zeros(self.num_tile, dtype=torch.int64)
        for i in range(self.num_tile):
            digit = perm[i].item()
            labels = MNIST_LABELS[digit]
            if self.random:
                self.index[i] = labels[torch.randint(0, len(labels), ())]
            else:
                # fix the digits for now as in Asai&Fukunaga 2017
                self.index[i] = labels[0]
            if digit == 0:
                self.location = [i // self.size, i % self.size]
        self.index = self.index.reshape(self.size, self.size)
        self.permutation = perm.reshape(self.size, self.size)
        return self.state()

    def _draw_with_replacement(self):
        init = torch.randint(1, self.num_class, (self.num_tile,))
        loc = torch.randint(0, self.num_tile, ())
        init[loc] = 0
        return init

    def state(self):
        canvas = torch.zeros(1, self.size*28, self.size*28)
        for i in range(self.size):
            for j in range(self.size):
                digit = MNIST_DATA[self.index[i, j]]
                canvas[0, i*28:(i+1)*28, j*28:(j+1)*28] = digit.clone()
        return canvas

    def goal_state(self):
        # only valid for 3x3 mnist permuted version
        canvas = torch.zeros(1, self.size*28, self.size*28)
        for i in range(self.size):
            for j in range(self.size):
                digit_idx = self.permutation[i, j]
                ii = digit_idx // self.size
                jj = digit_idx % self.size
                digit = MNIST_DATA[self.index[i, j]]
                canvas[0, ii*28:(ii+1)*28, jj*28:(jj+1)*28] = digit.clone()
        return canvas

    def random_goal_state(self):
        r = torch.randperm(self.num_tile)
        index = self.index.reshape(-1)[r]
        index = index.reshape(self.size, self.size)
        canvas = torch.zeros(1, self.size*28, self.size*28)
        for i in range(self.size):
            for j in range(self.size):
                digit = MNIST_DATA[index[i, j]]
                canvas[0, i*28:(i+1)*28, j*28:(j+1)*28] = digit.clone()
        return canvas

    def avg_goal_state(self, idx):
        # sigh...
        r = torch.randperm(self.num_tile)
        index = self.index.reshape(-1)[r]
        perm = self.permutation.reshape(-1)[r]
        zero_index, = torch.where(perm == 0)
        temp = index[zero_index].item()
        index[zero_index] = index[idx].item()
        index[idx] = temp

        index = index.reshape(self.size, self.size)
        canvas = torch.zeros(1, self.size*28, self.size*28)
        for i in range(self.size):
            for j in range(self.size):
                digit = MNIST_DATA[index[i, j]]
                canvas[0, i*28:(i+1)*28, j*28:(j+1)*28] = digit.clone()
        return canvas
