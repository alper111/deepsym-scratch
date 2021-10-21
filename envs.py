import torch

MNIST_DATA = torch.load("data/mnist_data.pt")
MNIST_LABELS = torch.load("data/mnist_label.pt")


class TilePuzzleMNIST:
    def __init__(self):
        self.action_names = [
            "move_right",
            "move_up",
            "move_left",
            "move_down"
        ],
        self.index = None
        self.location = None
        self.reset()
    
    def step(self, action):
        row, col = self.location
        blank_idx = self.index[row, col].item()
        if action == 0:
            if self.location[1] != 2:
                piece_idx = self.index[row, col+1].item()
                self.index[row, col] = piece_idx
                self.index[row, col+1] = blank_idx
                self.location[1] = col + 1
        elif action == 1:
            if self.location[0] != 0:
                piece_idx = self.index[row-1, col].item()
                self.index[row, col] = piece_idx
                self.index[row-1, col] = blank_idx
                self.location[0] = row - 1
        elif action == 2:
            if self.location[1] != 0:
                piece_idx = self.index[row, col-1].item()
                self.index[row, col] = piece_idx
                self.index[row, col-1] = blank_idx
                self.location[1] = col - 1
        elif action == 3:
            if self.location[0] != 2:
                piece_idx = self.index[row+1, col].item()
                self.index[row, col] = piece_idx
                self.index[row+1, col] = blank_idx
                self.location[0] = row + 1
        
        return self.state()
    
    def reset(self):
        perm = torch.randperm(9)
        self.index = torch.zeros(9, dtype=torch.int64)
        for i in range(9):
            digit = perm[i].item()
            labels = MNIST_LABELS[digit]
            self.index[i] = labels[torch.randint(0, len(labels), ())]
            if digit == 0:
                self.location = [i // 3, i % 3]
        self.index = self.index.reshape(3, 3)
        return self.state()

    def state(self):
        canvas = torch.zeros(1, 3*28, 3*28)
        for i in range(3):
            for j in range(3):
                digit = MNIST_DATA[self.index[i, j]]
                canvas[0, i*28:(i+1)*28, j*28:(j+1)*28] = digit.clone()
        return canvas
