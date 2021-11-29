import torch

MNIST_DATA = torch.load("data/mnist_data.pt")
MNIST_LABELS = torch.load("data/mnist_label.pt")


class TilePuzzleMNIST:
    def __init__(self, permutation=None):
        self.action_names = [
            "move_right",
            "move_up",
            "move_left",
            "move_down"
        ]
        self.permutation = None
        self.index = None
        self.location = None
        self.reset(permutation=permutation)

    def step(self, action):
        row, col = self.location
        blank_idx = self.index[row, col].item()
        if action == 0:
            if self.location[1] != 2:
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
            if self.location[0] != 2:
                piece_idx = self.index[row+1, col].item()
                self.index[row, col] = piece_idx
                self.index[row+1, col] = blank_idx
                self.location[0] = row + 1

                self.permutation[row, col] = self.permutation[row+1, col]
                self.permutation[row+1, col] = 0
        return self.state()
    
    def reset(self, permutation=None):
        if permutation is None:
            perm = torch.randperm(9)
        else:
            perm = permutation
        self.index = torch.zeros(9, dtype=torch.int64)
        for i in range(9):
            digit = perm[i].item()
            labels = MNIST_LABELS[digit]
            # self.index[i] = labels[torch.randint(0, len(labels), ())]
            # fix the digits for now as in Asai&Fukunaga 2017
            self.index[i] = labels[0]
            if digit == 0:
                self.location = [i // 3, i % 3]
        self.index = self.index.reshape(3, 3)
        self.permutation = perm.reshape(3, 3)
        return self.state()

    def state(self):
        canvas = torch.zeros(1, 3*28, 3*28)
        for i in range(3):
            for j in range(3):
                digit = MNIST_DATA[self.index[i, j]]
                canvas[0, i*28:(i+1)*28, j*28:(j+1)*28] = digit.clone()
        return canvas

    def goal_state(self):
        canvas = torch.zeros(1, 3*28, 3*28)
        for i in range(3):
            for j in range(3):
                digit_idx = self.permutation[i, j]
                ii = digit_idx // 3
                jj = digit_idx % 3
                digit = MNIST_DATA[self.index[i, j]]
                canvas[0, ii*28:(ii+1)*28, jj*28:(jj+1)*28] = digit.clone()
        return canvas

    def random_goal_state(self):
        randperm = torch.randperm(9)
        index = torch.zeros(9, dtype=torch.int64)
        for i in range(9):
            digit = randperm[i].item()
            labels = MNIST_LABELS[digit]
            # self.index[i] = labels[torch.randint(0, len(labels), ())]
            # fix the digits for now as in Asai&Fukunaga 2017
            index[i] = labels[0]
        index = index.reshape(3, 3)
        canvas = torch.zeros(1, 3*28, 3*28)
        for i in range(3):
            for j in range(3):
                digit = MNIST_DATA[index[i, j]]
                canvas[0, i*28:(i+1)*28, j*28:(j+1)*28] = digit.clone()
        return canvas
