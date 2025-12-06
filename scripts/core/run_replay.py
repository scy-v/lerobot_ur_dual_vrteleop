import time
import yaml
import logging
logging.basicConfig(level=logging.WARNING, format="%(message)s")
from pathlib import Path
from typing import Dict, Any
from lerobot_robot_ur import URConfig, UR
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.robot_utils import busy_wait
from lerobot.utils.utils import log_say

class ReplayConfig:
    def __init__(self, cfg: Dict[str, Any]):
        robot = cfg["robot"]

        # global config
        self.dataset_name: str = cfg["dataset_name"]
        self.episode_idx: str = cfg.get("episode_idx", 0)
        self.debug: bool = cfg.get("debug", False)

        # robot config
        self.left_robot_ip: str = robot["left_robot_ip"]
        self.right_robot_ip: str = robot["right_robot_ip"]
        self.left_gripper_port: str = robot["left_gripper_port"]
        self.right_gripper_port: str = robot["right_gripper_port"]
        self.use_gripper: bool = robot["use_gripper"]
        self.gripper_reverse: bool = robot["gripper_reverse"]

def run_replay(replay_cfg: ReplayConfig):
    episode_idx = replay_cfg.episode_idx

    robot_config = URConfig(
        left_robot_ip=replay_cfg.left_robot_ip,
        right_robot_ip=replay_cfg.right_robot_ip,
        left_gripper_port=replay_cfg.left_gripper_port,
        right_gripper_port=replay_cfg.right_gripper_port,
        use_gripper=replay_cfg.use_gripper,
        debug=replay_cfg.debug,
        gripper_reverse=replay_cfg.gripper_reverse
    )

    robot = UR(robot_config)
    robot.connect()
    dataset = LeRobotDataset(replay_cfg.dataset_name, episodes=[episode_idx])
    actions = dataset.hf_dataset.select_columns("action")
    logging.info(f"Replaying episode {episode_idx}")

    for idx in range(dataset.num_frames):
        t0 = time.perf_counter()

        action = {
            name: float(actions[idx]["action"][i]) for i, name in enumerate(dataset.features["action"]["names"])
        }

        robot.send_action(action)

        busy_wait(1.0 / dataset.fps - (time.perf_counter() - t0))

    robot.disconnect()

def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    replay_cfg = ReplayConfig(cfg["replay"])

    run_replay(replay_cfg)