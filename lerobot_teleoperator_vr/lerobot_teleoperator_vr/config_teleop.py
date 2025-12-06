from dataclasses import dataclass
from xrobotoolkit_teleop.common.xr_client import XrClient
from lerobot.teleoperators.config import TeleoperatorConfig


@TeleoperatorConfig.register_subclass("lerobot_teleoperator_vr")
@dataclass
class VRTeleopConfig(TeleoperatorConfig):
    fps: str
    left_robot_ip: str
    right_robot_ip: str
    xr_client: XrClient
    trigger_reverse: bool
    trigger_threshold: float
    close_position: float
    open_position: float
    servo_time: float
    scale_factor: float
    robot_urdf_path: str
    R_headset_world: list[float]
    visualize_placo: bool = False
    control_mode: str = "vrteleop"
