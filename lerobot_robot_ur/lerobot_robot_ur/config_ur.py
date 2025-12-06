from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from lerobot.robots.config import RobotConfig

@RobotConfig.register_subclass("ur_robot")
@dataclass
class URConfig(RobotConfig):
    use_gripper: bool = True
    gripper_reverse: bool = True
    left_robot_ip: str = "192.168.131.11"
    right_robot_ip: str = "192.168.131.12"
    gain: float = 300
    servo_time: float = 0.017
    lookahead_time: float = 0.1
    left_gripper_port: str = "/dev/ur_left_gripper"
    right_gripper_port: str = "/dev/ur_right_gripper"
    gripper_bin_threshold: float = 0.98
    debug: bool = True
    close_threshold: float = 0.7
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
