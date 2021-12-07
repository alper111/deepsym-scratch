import time

import numpy as np
from scipy.spatial.transform import Rotation

import comm


def get_rgbd(env):
    rgb = env.get_rgb()
    d = env.get_depth()
    d = d.reshape(1, d.shape[0], d.shape[1])
    rgbd = (np.concatenate((rgb, d), axis=0)*255).astype(np.uint8)
    return rgbd


env = comm.Communication(20000)
env.open_connection()
env.start()


scales = np.linspace(1.0, 2.0, 10)
depth_prev = []
depth_next = []
rgb_prev = []
rgb_next = []

for obj_i in range(5):
    for s_i in scales:
        for obj_j in [4]:
            for s_j in scales:
                start = time.time()
                print(obj_i, s_i, obj_j, s_j)
                size_i = s_i * 0.1
                size_j = s_j * 0.1
                loc_i = [-0.75, -0.15, 0.7+size_i/2]
                loc_j = [-0.75, 0.16, 0.7+size_j/2]
                env.generate_object(obj_i, loc_i)
                env.set_object_scale(env.generated_objects[-1], s_i, s_i, s_i)
                env.step()

                env.generate_object(obj_j, loc_j)
                env.set_object_scale(env.generated_objects[-1], s_j, s_j, s_j)
                env.step()

                depth_prev.append(env.get_depth())
                rgb_prev.append(env.get_rgb())
                if obj_j == 3:
                    quat = Rotation.from_euler("z", 90, degrees=True).as_quat().tolist()
                else:
                    quat = [0., 0., 0., 1.]

                env.hand_open_pose()
                env.set_tip_pose([-0.75, 0.16, 1.0], quaternion=quat)
                if obj_j in [1, 2, 4]:
                    multiplier = 0.95
                else:
                    multiplier = 0.75
                env.set_tip_pose([-0.75, 0.16, 0.7+multiplier*size_j], quaternion=quat)
                env.hand_grasp_pose()
                env.set_tip_pose([-0.75, 0.16, 0.7+size_i+size_j+0.05], quaternion=quat)
                env.set_tip_pose([-0.75, -0.15, 0.7+size_i+size_j+0.05], quaternion=quat)
                env.hand_open_pose()
                
                depth_next.append(env.get_depth())
                rgb_next.append(env.get_rgb())
                env.set_tip_pose([-0.3, -0.1139, 1.1856], wait=5)
                # env.init_arm_pose()
                env.remove_object(env.generated_objects[-1])
                env.remove_object(env.generated_objects[-1])

                end = time.time()
                print(end-start)

np.save("data/exploration_second/depth_prev.npy", depth_prev)
np.save("data/exploration_second/depth_next.npy", depth_next)
np.save("data/exploration_second/rgb_prev.npy", rgb_prev)
np.save("data/exploration_second/rgb_next.npy", rgb_next)

print("simulation stopped.")
