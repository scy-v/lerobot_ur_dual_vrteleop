from pyDHgripper import PGE
from pathlib import Path
import yaml
import tyro

def get_vel(gripper):
    return gripper.write_uart(modbus_high_addr=0x01,
                          modbus_low_addr=0x04,
                          is_set=False)

def run(type: str = "right"):
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)
    left_gripper_port = cfg["record"]["robot"]["gripper"]["left_gripper_port"]
    right_gripper_port = cfg["record"]["robot"]["gripper"]["right_gripper_port"]
    if type == "left":
        gripper = PGE(left_gripper_port)
    else:
        gripper = PGE(right_gripper_port)
    gripper.init_feedback()
    gripper.set_force(20)
    gripper.set_vel(100)

    while True:
        val = input("enter: ")
        gripper.set_pos(val=int(val), blocking=False)

def main():
    tyro.cli(run)


