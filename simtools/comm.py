import time

import numpy as np
from scipy.spatial.transform import Rotation

import sim


class Communication:
    def __init__(self, port):
        self.port = port
        self.client_id = None
        self.rgb_cam = None
        self.depth_cam = None
        self.target_handle = None
        self.tip_handle = None
        self.force_handle = None
        self.joint_handles = []
        self.gripper_handles = []
        self.generated_objects = []
        self.object_types = ["sphere", "cube", "cylinder", "cylinder_side", "hollow"]

    def open_connection(self):
        while self.client_id != 0:
            self.client_id = sim.simxStart("127.0.0.1", self.port, True, True, 5000, 1)
            if self.client_id != 0:
                print(f"client id: {self.client_id} for port {self.port}. Retrying in 5 secs")
                time.sleep(5)
        sim.simxSynchronous(self.client_id, True)

    def close_connection(self):
        sim.simxFinish(self.client_id)

    def start(self):
        self._initialize_handles()
        sim.simxStartSimulation(self.client_id, sim.simx_opmode_blocking)

    def stop(self):
        self.rgb_cam = None
        self.depth_cam = None
        self.joint_handles = []
        self.gripper_handles = []
        self.target_handle = None
        self.tip_handle = None
        self.force_handle = None
        self.generated_objects = []
        sim.simxStopSimulation(self.client_id, sim.simx_opmode_blocking)

    def load_scene(self, path):
        sim.simxLoadScene(self.client_id, path, 1, sim.simx_opmode_blocking)

    def set_tip_pose(self, position, quaternion=[0., 0., 0., 1.], wait=20):
        pose = self.get_tip_pose()
        curr_position, curr_quaternion = pose[:3], pose[3:]
        curr_orientation = Rotation.from_quat(curr_quaternion).as_euler("zyx")
        orientation = Rotation.from_quat(quaternion).as_euler("zyx")
        position_traj = np.linspace(curr_position, position, wait+1)
        orientation_traj = np.linspace(curr_orientation, orientation, wait+1)
        quaternion_traj = Rotation.from_euler("zyx", orientation_traj).as_quat()
        for i in range(1, wait+1):
            self.set_object_position(self.target_handle, position_traj[i])
            self.set_object_quaternion(self.target_handle, quaternion_traj[i])
            self.step()

    def set_joint_position(self, position, wait=20):
        for i in range(6):
            sim.simxSetJointPosition(self.client_id, self.joint_handles[i], position[i], sim.simx_opmode_oneshot)
        self._wait(wait)

    def set_gripper_position(self, position, wait=True):
        for i in range(3):
            for j in range(3):
                handle = self.gripper_handles[i][j]
                if handle != -1:
                    sim.simxSetJointTargetPosition(self.client_id, handle, position[i][j], sim.simx_opmode_oneshot)
        self._wait(wait)

    def set_object_position(self, obj_id, position):
        sim.simxSetObjectPosition(self.client_id, obj_id, -1, position, sim.simx_opmode_oneshot)

    def set_object_quaternion(self, obj_id, quaternion):
        sim.simxSetObjectQuaternion(self.client_id, obj_id, -1, quaternion, sim.simx_opmode_oneshot)

    def set_object_scale(self, obj_id, xs, ys, zs):
        sim.simxSetStringSignal(self.client_id, "set_scale", f"{obj_id}#{xs}#{ys}#{zs}", sim.simx_opmode_oneshot)

    def set_object_color(self, obj_id, r, g, b):
        sim.simxSetStringSignal(self.client_id, "set_color", f"{obj_id}#{r}#{g}#{b}", sim.simx_opmode_oneshot)

    def generate_object(self, obj_type, position, quaternion=[0., 0., 0., 1.]):
        handle = self._get_object_handle(self.object_types[obj_type])
        code = -1
        it = 0
        while code != 0:
            code, (generated_object,) = sim.simxCopyPasteObjects(self.client_id, [handle], sim.simx_opmode_blocking)
            if code != 0 and it == 0:
                print(f"Code is {code} for copypasteobjects")
            it += 1
        self.generated_objects.append(generated_object)
        self.set_object_position(generated_object, position)
        self.set_object_quaternion(generated_object, quaternion)

    def remove_object(self, obj_id):
        sim.simxRemoveObject(self.client_id, obj_id, sim.simx_opmode_blocking)
        self.generated_objects.remove(obj_id)

    def get_tip_pose(self):
        code, position = sim.simxGetObjectPosition(self.client_id, self.tip_handle, -1, sim.simx_opmode_blocking)
        code, quaternion = sim.simxGetObjectQuaternion(self.client_id, self.tip_handle, -1, sim.simx_opmode_blocking)
        return position + quaternion

    def get_joint_position(self):
        position = []
        for i in range(6):
            code, p_i = sim.simxGetJointPosition(self.client_id, self.joint_handles[i], sim.simx_opmode_blocking)
            position.append(p_i)
        return position

    def get_gripper_position(self):
        position = []
        for i in range(3):
            finger_position = []
            for j in range(3):
                handle = self.gripper_handles[i][j]
                if handle != -1:
                    p_ij = sim.simxGetJointPosition(self.client_id, handle, sim.simx_opmode_blocking)
                    finger_position.append(p_ij)
                else:
                    finger_position.append(-1)
            position.append(finger_position)
        return position

    def get_object_poses(self):
        poses = []
        for obj in self.generated_objects:
            code, position = sim.simxGetObjectPosition(self.client_id, obj, -1, sim.simx_opmode_blocking)
            code, quaternion = sim.simxGetObjectQuaternion(self.client_id, obj, -1, sim.simx_opmode_blocking)
            poses.append(position + quaternion)
        return poses

    def get_rgb(self):
        code, res, image = sim.simxGetVisionSensorImage(self.client_id, self.rgb_cam, 0, sim.simx_opmode_blocking)
        image = np.array(image, dtype=np.uint8).reshape(res[1], res[0], 3)[::-1] / 255.0
        image = np.transpose(image, (2, 0, 1))
        return image

    def get_depth(self):
        code, res, image = sim.simxGetVisionSensorDepthBuffer(self.client_id, self.depth_cam, sim.simx_opmode_blocking)
        image = np.array(image).reshape(res[1], res[0])[::-1].copy()
        # you should change this in the next experimentation!
        image = image[8:120, 8:120]
        return image

    def get_force_sensor(self):
        code, state, force, torque = sim.simxReadForceSensor(self.client_id, self.force_handle, sim.simx_opmode_blocking)
        return force

    def step(self):
        sim.simxSynchronousTrigger(self.client_id)
        sim.simxGetPingTime(self.client_id)

    def init_arm_pose(self, wait=20):
        self.set_tip_pose([-0.2, -0.1139, 1.1858], wait=wait)

    def hand_grasp_pose(self, wait=20):
        position = np.array([
            [-90, 0, -90],
            [180, 180, 180],
            [60, 60, 60]
        ])
        position = np.radians(position)
        self.set_gripper_position(position, wait)

    def hand_open_pose(self, wait=20):
        position = np.array([
            [-90, 0, -90],
            [0, 0, 0],
            [45, 45, 45]
        ])
        position = np.radians(position)
        self.set_gripper_position(position, wait)

    def hand_poke_pose(self, wait=20):
        position = np.array([
            [-90, 0, -90],
            [90, 90, 90],
            [90, -5.0, 90]
        ])
        position = np.radians(position)
        self.set_gripper_position(position, wait)

    def hand_fist_pose(self, wait=20):
        position = np.array([
            [-90, 0, -90],
            [85, 85, 85],
            [90, 90, 90]
        ])
        position = np.radians(position)
        self.set_gripper_position(position, wait)

    def _initialize_handles(self):
        self.rgb_cam = self._get_object_handle("kinect_rgb")
        self.depth_cam = self._get_object_handle("kinect_depth")
        self.target_handle = self._get_object_handle("target")
        self.tip_handle = self._get_object_handle("tip")
        self.force_handle = self._get_object_handle("UR10_connection")
        for i in range(1, 7):
            self.joint_handles.append(self._get_object_handle("UR10_joint%d" % i))

        for f in ["A", "B", "C"]:
            finger_handles = []
            for i in range(3):
                if i == 1 and f == "A":
                    finger_handles.append(-1)
                else:
                    finger_handles.append(self._get_object_handle("BarrettHand_joint%s_%d" % (f, i)))
            self.gripper_handles.append(finger_handles)
        self._set_gripper_mode(1)

    def _set_gripper_mode(self, mode):
        for finger in self.gripper_handles:
            for joint in finger:
                if joint != -1:
                    sim.simxSetObjectIntParameter(self.client_id, joint, sim.sim_jointintparam_ctrl_enabled,
                                                  mode, sim.simx_opmode_blocking)
                    sim.simxSetObjectIntParameter(self.client_id, joint, sim.sim_jointintparam_velocity_lock,
                                                  1-mode, sim.simx_opmode_blocking)

    def _get_object_handle(self, object_name):
        _, handle = sim.simxGetObjectHandle(self.client_id, object_name, sim.simx_opmode_blocking)
        return handle

    def _wait(self, step=20):
        for i in range(step):
            self.step()
