import logging
import time
from typing import Any
import threading
from rtde_control import RTDEControlInterface
from rtde_receive import RTDEReceiveInterface

from lerobot.cameras import make_cameras_from_configs
from lerobot.utils.errors import DeviceNotConnectedError, DeviceAlreadyConnectedError
from lerobot.robots.robot import Robot
from pyDHgripper import PGE
from .config_ur import URConfig

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
class UR(Robot):
    config_class = URConfig
    name = "ur"

    def __init__(self, config: URConfig):
        super().__init__(config)
        self.cameras = make_cameras_from_configs(config.cameras)

        self.cfg = config
        self._is_connected = False
        self.arm = {}
        self._gripper = None
        self._initial_pose = None
        self._prev_observation = None
        self._num_joints = 6
        self._gripper_force = 20
        self._left_gripper_position = 1
        self._right_gripper_position = 1
        self._velocity = 0.5 # not used in current version
        self._acceleration = 0.5 # not used in current version
        self._left_last_gripper_position = 1
        self._right_last_gripper_position = 1
        self.stop_signal = threading.Event()
        
    def connect(self) -> None:
        if self.is_connected:
            raise DeviceAlreadyConnectedError(f"{self.name} is already connected.")

        # Connect to robot
        self.arm['left_rtde_r'], self.arm['left_rtde_c'] = self._check_ur_connection(self.cfg.left_robot_ip, "left")
        self.arm['right_rtde_r'], self.arm['right_rtde_c'] = self._check_ur_connection(self.cfg.right_robot_ip, "right")

        # Initialize gripper
        if self.cfg.use_gripper:
            self._left_gripper = self._check_gripper_connection(self.cfg.left_gripper_port, "left")
            self._right_gripper = self._check_gripper_connection(self.cfg.right_gripper_port, "right")

            # Start gripper state reader
            self._start_update_gripper_state()

        # Connect cameras
        logger.info("\n===== [CAM] Initializing Cameras =====")
        for cam_name, cam in self.cameras.items():
            cam.connect()
            logger.info(f"[CAM] {cam_name} connected successfully.")
        logger.info("===== [CAM] Cameras Initialized Successfully =====\n")

        self.is_connected = True
        logger.info(f"[INFO] {self.name} env initialization completed successfully.\n")

    def _check_gripper_connection(self, port: str, gripper_name: str):
        logger.info(f"\n===== [GRIPPER] Initializing {gripper_name} gripper...")
        gripper = PGE(port)
        gripper.init_feedback()
        gripper.set_force(self._gripper_force)
        logger.info("===== [GRIPPER] Gripper initialized successfully.\n")
        return gripper


    def _check_ur_connection(self, robot_ip: str, arm_name: str):
        try:
            logger.info(f"\n===== [ROBOT] Connecting to {arm_name} UR robot =====")
            rtde_r = RTDEReceiveInterface(robot_ip)
            rtde_c = RTDEControlInterface(robot_ip)

            joint_positions = rtde_r.getActualQ()
            if joint_positions is not None and len(joint_positions) == self._num_joints:
                formatted_joints = [round(j, 4) for j in joint_positions]
                logger.info(f"[ROBOT] Current {arm_name} UR joint positions: {formatted_joints}")
                logger.info(f"===== [ROBOT] {arm_name} UR connected successfully =====\n")
            else:
                logger.info(f"===== [ERROR] Failed to read {arm_name} UR joint positions. Check connection or remote control mode =====")

        except Exception as e:
            logger.info(f"===== [ERROR] Failed to connect to {arm_name} UR robot =====")
            logger.info(f"Exception: {e}\n")

        return rtde_r, rtde_c

    def _start_update_gripper_state(self):
        threading.Thread(target=self._update_left_gripper_state, daemon=True).start()
        threading.Thread(target=self._update_right_gripper_state, daemon=True).start()

    def _update_left_gripper_state(self):
        self._left_gripper.pos = None
        while not self.stop_signal.is_set():
            gripper_position = 0.0 if self._left_gripper_position  < self.cfg.close_threshold else 1.0
            if self.cfg.gripper_reverse:
                gripper_position = 1 - gripper_position

            if gripper_position != self._left_last_gripper_position:
                for _ in range(4):  
                    self._left_gripper.set_pos(val=int(1000 * gripper_position), blocking=False)
                self._left_last_gripper_position = gripper_position

            gripper_pos = self._left_gripper.read_pos() / 1000.0
            if self.cfg.gripper_reverse:
                gripper_pos = 1 - gripper_pos

            self._left_gripper.pos = gripper_pos
            time.sleep(0.01)

    def _update_right_gripper_state(self):
        self._right_gripper.pos = None
        while not self.stop_signal.is_set():
            gripper_position = 0.0 if self._right_gripper_position  < self.cfg.close_threshold else 1.0
            if self.cfg.gripper_reverse:
                gripper_position = 1 - gripper_position

            if gripper_position != self._right_last_gripper_position:
                for _ in range(4):
                    self._right_gripper.set_pos(val=int(1000 * gripper_position), blocking=False) 
                self._right_last_gripper_position = gripper_position

            gripper_pos = self._right_gripper.read_pos() / 1000.0
            if self.cfg.gripper_reverse:
                gripper_pos = 1 - gripper_pos

            self._right_gripper.pos = gripper_pos
            time.sleep(0.01)

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            **{f"left_joint_{i+1}.pos": float for i in range(self._num_joints)}, # left joint positions
            **{"left_gripper_raw_position": float}, # left gripper raw position
            **{f"right_joint_{i+1}.pos": float for i in range(self._num_joints)}, # right joint positions
            **{"right_gripper_raw_position": float}, # right gripper raw position
            **{"left_gripper_raw_bin": float}, # other gripper state: left gripper raw position bin
            **{"left_gripper_action_bin": float}, # other gripper state: left gripper action command bin
            **{"right_gripper_raw_bin": float}, # other gripper state: right gripper raw position bin
            **{"right_gripper_action_bin": float}, # other gripper state: right gripper action command bin
            **{f"left_joint_{i+1}.vel": float for i in range(self._num_joints)}, # left joint velocities
            **{f"right_joint_{i+1}.vel": float for i in range(self._num_joints)}, # right joint velocities
            **{f"left_joint_{i+1}.acc": float for i in range(self._num_joints)}, # left joint accelerations
            **{f"right_joint_{i+1}.acc": float for i in range(self._num_joints)}, # right joint accelerations
            **{f"left_joint_{i+1}.frc": float for i in range(6)}, # left joint forces
            **{f"right_joint_{i+1}.frc": float for i in range(6)}, # right joint forces
            **{f"left_tcp_pose.{axis}": float for axis in ["x", "y", "z","rx","ry","rz"]}, # left tcp pose
            **{f"right_tcp_pose.{axis}": float for axis in ["x", "y", "z","rx","ry","rz"]}, # right tcp pose
            **{f"left_tcp_vel.{axis}": float for axis in ["x", "y", "z","rx","ry","rz"]}, # left tcp velocity
            **{f"right_tcp_vel.{axis}": float for axis in ["x", "y", "z","rx","ry","rz"]}, # right tcp velocity
            **{f"left_tcp_acc.{axis}": float for axis in ["x", "y", "z"]}, # left tcp acceleration
            **{f"right_tcp_acc.{axis}": float for axis in ["x", "y", "z"]}, # right tcp acceleration
            **{f"left_tcp_frc.{axis}": float for axis in ["x", "y", "z","rx","ry","rz"]}, # left tcp force
            **{f"right_tcp_frc.{axis}": float for axis in ["x", "y", "z","rx","ry","rz"]}, # right tcp force
        }

    @property
    def action_features(self) -> dict[str, type]:
        return {
            **{f"left_joint_{i+1}.pos": float for i in range(self._num_joints)}, # left joint positions
            **{"left_gripper_position": float}, # left gripper position
            **{f"right_joint_{i+1}.pos": float for i in range(self._num_joints)}, # right joint positions
            **{"right_gripper_position": float}, # right gripper position
        }

    def send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")

        joint_positions = [
            action[f"{arm}_joint_{i+1}.pos"] for arm in ["left", "right"] for i in range(self._num_joints)
        ]
        if not self.cfg.debug:       
            t_start = self.arm["left_rtde_c"].initPeriod()
            self.arm["left_rtde_c"].servoJ(joint_positions[:self._num_joints], self._velocity, self._acceleration, self.cfg.servo_time, self.cfg.lookahead_time, self.cfg.gain)
            self.arm["left_rtde_c"].waitPeriod(t_start)

            t_start = self.arm["right_rtde_c"].initPeriod()
            self.arm["right_rtde_c"].servoJ(joint_positions[self._num_joints:], self._velocity, self._acceleration, self.cfg.servo_time, self.cfg.lookahead_time, self.cfg.gain)
            self.arm["right_rtde_c"].waitPeriod(t_start)

        if "left_gripper_position" in action and "right_gripper_position" in action and self.cfg.use_gripper:
            self._left_gripper_position = action["left_gripper_position"]
            self._right_gripper_position = action["right_gripper_position"]

        return action

    def get_observation(self) -> dict[str, Any]:
        if not self.is_connected:
            raise DeviceNotConnectedError(f"{self} is not connected.")
        
        # Read joint positions
        left_qpos = self.arm["left_rtde_r"].getActualQ()

        right_qpos = self.arm["right_rtde_r"].getActualQ()
        # Read joint velocities
        left_qd = self.arm["left_rtde_r"].getActualQd()
        right_qd = self.arm["right_rtde_r"].getActualQd()

        # Read joint accelerations
        left_qdd = self.arm["left_rtde_r"].getTargetQdd()
        right_qdd = self.arm["right_rtde_r"].getTargetQdd()

        # Read joint forces
        left_qfrc = self.arm["left_rtde_c"].getJointTorques()
        right_qfrc = self.arm["right_rtde_c"].getJointTorques()

        # Read tcp pose
        left_tcp_pose = self.arm["left_rtde_r"].getActualTCPPose()
        right_tcp_pose = self.arm["right_rtde_r"].getActualTCPPose()

        # Read tcp speed
        left_tcp_vel = self.arm["left_rtde_r"].getActualTCPSpeed()
        right_tcp_vel = self.arm["right_rtde_r"].getActualTCPSpeed()

        # Read tcp acceleration
        left_tcp_acc = self.arm["left_rtde_r"].getActualToolAccelerometer()
        right_tcp_acc = self.arm["right_rtde_r"].getActualToolAccelerometer()

        # Read tcp force
        left_tcp_frc = self.arm["left_rtde_r"].getActualTCPForce()
        right_tcp_frc = self.arm["right_rtde_r"].getActualTCPForce()

        # Prepare observation dictionary
        obs_dict = {}

        for i in range(len(left_qpos)):
            obs_dict[f"left_joint_{i+1}.pos"] = left_qpos[i]
            obs_dict[f"left_joint_{i+1}.vel"] = left_qd[i]
            obs_dict[f"left_joint_{i+1}.acc"] = left_qdd[i]
            obs_dict[f"left_joint_{i+1}.frc"] = left_qfrc[i]

        for i in range(len(right_qpos)):
            obs_dict[f"right_joint_{i+1}.pos"] = right_qpos[i]
            obs_dict[f"right_joint_{i+1}.vel"] = right_qd[i]
            obs_dict[f"right_joint_{i+1}.acc"] = right_qdd[i]
            obs_dict[f"right_joint_{i+1}.frc"] = right_qfrc[i]

        for i, axis in enumerate(["x", "y", "z","rx","ry","rz"]):
            obs_dict[f"left_tcp_pose.{axis}"] = left_tcp_pose[i]
            obs_dict[f"right_tcp_pose.{axis}"] = right_tcp_pose[i]
            obs_dict[f"left_tcp_vel.{axis}"] = left_tcp_vel[i]
            obs_dict[f"right_tcp_vel.{axis}"] = right_tcp_vel[i]
            if i < 3: # tcp_acceleration have only 3 axes
                obs_dict[f"left_tcp_acc.{axis}"] = left_tcp_acc[i]
                obs_dict[f"right_tcp_acc.{axis}"] = right_tcp_acc[i]
            obs_dict[f"left_tcp_frc.{axis}"] = left_tcp_frc[i]
            obs_dict[f"right_tcp_frc.{axis}"] = right_tcp_frc[i]

        if self.cfg.use_gripper:
            obs_dict["left_gripper_raw_position"] = self._left_gripper.pos
            obs_dict["left_gripper_action_bin"] = self._left_last_gripper_position
            obs_dict["left_gripper_raw_bin"] = 0 if self._left_gripper.pos <= self.cfg.gripper_bin_threshold else 1
            obs_dict["right_gripper_raw_position"] = self._right_gripper.pos
            obs_dict["right_gripper_action_bin"] = self._right_last_gripper_position
            obs_dict["right_gripper_raw_bin"] = 0 if self._right_gripper.pos <= self.cfg.gripper_bin_threshold else 1
        else:
            obs_dict["left_gripper_raw_position"] = None
            obs_dict["left_gripper_action_bin"] = None
            obs_dict["left_gripper_raw_bin"] = None
            obs_dict["right_gripper_raw_position"] = None
            obs_dict["right_gripper_action_bin"] = None
            obs_dict["right_gripper_raw_bin"] = None

        # Capture images from cameras
        for cam_key, cam in self.cameras.items():
            start = time.perf_counter()
            obs_dict[cam_key] = cam.read()
            dt_ms = (time.perf_counter() - start) * 1e3
            logger.debug(f"{self} read {cam_key}: {dt_ms:.1f}ms")

        self._prev_observation = obs_dict

        return obs_dict

    def disconnect(self) -> None:
        if not self.is_connected:
            return

        if self.arm is not None:
            self.arm["left_rtde_c"].disconnect()
            self.arm["left_rtde_r"].disconnect()
            self.arm["right_rtde_r"].disconnect()
            self.arm["right_rtde_r"].disconnect()

        for cam in self.cameras.values():
            cam.disconnect()

        self.stop_signal.set()
        self.is_connected = False
        logger.info(f"[INFO] ===== All {self.name} connections have been closed =====")

    def calibrate(self) -> None:
        pass

    def is_calibrated(self) -> bool:
        return self.is_connected
    
    def configure(self) -> None:
        pass

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @is_connected.setter
    def is_connected(self, value: bool) -> None:
        self._is_connected = value

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        return {
           cam: (self.cameras[cam].height, self.cameras[cam].width, 3) for cam in self.cameras
        }

    @property
    def observation_features(self) -> dict[str, Any]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def cameras(self):
        return self._cameras

    @cameras.setter
    def cameras(self, value):
        self._cameras = value

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, value):
        self._config = value
