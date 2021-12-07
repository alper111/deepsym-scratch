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
xrng = np.linspace(-0.4, -1.1, 5)[:4]
yrng = np.linspace(-0.35, 0.35, 5)[:4]
state_prev = []
state_next = []
pos_prev = []
pos_next = []
force_readings = []
action = []

for obj_i in range(5):
    for scale in reversed(scales):
        for x in xrng:
            for y in yrng:
                for a in range(3):
                    print(obj_i, scale, a, x, y)
                    env.set_tip_pose([-0.3, -0.1139, 1.1856], wait=5)
                    size = 0.1 * scale
                    position = [x, y, 0.7+size/2]
                    env.generate_object(obj_i, position)
                    env.set_object_scale(env.generated_objects[0], scale, scale, scale)
                    env.set_object_color(env.generated_objects[0], 1.0, 0.35, 0.35)
                    env.step()

                    state_prev.append(get_rgbd(env))
                    pos_prev.append(env.get_object_poses())
                    # push top
                    if a == 0:
                        env.hand_poke_pose()
                        # env.hand_fist_pose()
                        env.set_tip_pose([x-0.05, y, 1.0], wait=20)
                        env.set_tip_pose([x-0.05, y, 0.7725+size])
                        force_readings.append(env.get_force_sensor())
                        env.set_tip_pose([x-0.05, y, 1.0], wait=5)
                    # push front
                    elif a == 1:
                        env.hand_fist_pose()
                        env.set_tip_pose([x+0.175, y, 1.0], wait=10)
                        env.set_tip_pose([x+0.175, y, 0.68+size/2])
                        env.set_tip_pose([x-0.05, y, 0.68+size/2])
                        force_readings.append(env.get_force_sensor())
                        env.set_tip_pose([x+0.175, y, 0.68+size/2], wait=5)
                    # push side
                    elif a == 2:
                        env.hand_fist_pose()
                        quat = Rotation.from_euler("z", 270, degrees=True).as_quat().tolist()
                        env.set_tip_pose([x, y-0.175-size/2, 1.0], quaternion=quat, wait=10)
                        env.set_tip_pose([x, y-0.175-size/2, 0.68+size/2], quaternion=quat)
                        env.set_tip_pose([x, y+0.05, 0.68+size/2], quaternion=quat)
                        force_readings.append(env.get_force_sensor())
                        env.set_tip_pose([x, y-0.175-size/2, 0.68+size/2], quaternion=quat, wait=5)

                    state_next.append(get_rgbd(env))
                    pos_next.append(env.get_object_poses())
                    action.append(a)
                    env.set_tip_pose([-0.3, -0.1139, 1.1856], wait=5)
                    env.remove_object(env.generated_objects[-1])
                    print(force_readings[-1])
                    
np.save("data/exploration_first/obs_prev.npy", state_prev)
np.save("data/exploration_first/obs_next.npy", state_next)
np.save("data/exploration_first/pos_prev.npy", pos_prev)
np.save("data/exploration_first/pos_next.npy", pos_next)
np.save("data/exploration_first/force_readings.npy", force_readings)
np.save("data/exploration_first/actions.npy", action)
