import socket
import rtde_control
import rtde_receive
from pathlib import Path
import yaml

class Robot_Config:
    def __init__(self, cfg):
        robot = cfg["robot"]
        self.left_robot_ip: str = robot["left_robot_ip"]
        self.right_robot_ip: str = robot["right_robot_ip"]

def ur_freedrive(cfg):
    # UR IP
    left_ip = cfg.left_robot_ip
    right_ip = cfg.right_robot_ip

    # Connect to RTDE Control Interface
    left_rtde_c = rtde_control.RTDEControlInterface(left_ip)
    right_rtde_c = rtde_control.RTDEControlInterface(right_ip)

    try:
        # Enter Freedrive Mode
        left_rtde_c.freedriveMode()
        right_rtde_c.freedriveMode()

        input("Press Enter to exit freedrive mode...")

    except Exception as e:
        print(f"Exception occurred: {e}")

    except KeyboardInterrupt:
        print("Freedrive interrupted by Ctrl+C.")

    finally:
        left_rtde_c.endFreedriveMode()
        right_rtde_c.endFreedriveMode()
        print("Both arms exited freedrive mode.")

def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    robot_cfg = Robot_Config(cfg["record"])
    ur_freedrive(robot_cfg)