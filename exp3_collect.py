"""
Collect MNIST 8-tile experience.
"""
import torch
from envs import TilePuzzleMNIST

NUM_TRAJ = 100000
EPISODE_LENGTH = 1
SIZE = 3
NUM_ACTIONS = 4

state = torch.zeros(NUM_TRAJ, EPISODE_LENGTH, 1, SIZE*28, SIZE*28, dtype=torch.uint8)
effect = torch.zeros(NUM_TRAJ, EPISODE_LENGTH, 1, SIZE*28, SIZE*28, dtype=torch.int16)
action = torch.zeros(NUM_TRAJ, EPISODE_LENGTH, NUM_ACTIONS, dtype=torch.uint8)
eye = torch.eye(NUM_ACTIONS, dtype=torch.uint8)

for traj in range(NUM_TRAJ):
    env = TilePuzzleMNIST(size=SIZE)
    for t in range(EPISODE_LENGTH):
        s_before = env.state()
        action_index = torch.randint(0, NUM_ACTIONS, ())
        s_after = env.step(action_index)

        state[traj, t] = (s_before*255).type(torch.uint8)
        effect[traj, t] = ((s_after - s_before)*255).type(torch.int16)
        action[traj, t] = eye[action_index]
    print(f"{traj+1} completed")

state = state.reshape(-1, 1, SIZE*28, SIZE*28)
effect = effect.reshape(-1, 1, SIZE*28, SIZE*28)
action = action.reshape(-1, NUM_ACTIONS)

torch.save(state, "data/tile_state.pt")
torch.save(effect, "data/tile_effect.pt")
torch.save(action, "data/tile_action.pt")
