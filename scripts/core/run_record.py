import yaml
from pathlib import Path
from typing import Dict, Any
from scripts.utils.dataset_utils import generate_dataset_name, update_dataset_info
from lerobot_robot_ur import URConfig, UR
from lerobot_teleoperator_vr import VRTeleopConfig, VRTeleop
from lerobot.cameras.configs import ColorMode, Cv2Rotation
from lerobot.cameras.realsense.camera_realsense import RealSenseCameraConfig
from lerobot.scripts.lerobot_record import record_loop
from lerobot.processor import make_default_processors
from lerobot.utils.visualization_utils import init_rerun
from lerobot.utils.control_utils import init_keyboard_listener
import shutil
import time
import signal
import os
import threading
import termios, sys
from xrobotoolkit_teleop.common.xr_client import XrClient
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.utils.control_utils import sanity_check_dataset_robot_compatibility
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")

class RecordConfig:
    def __init__(self, cfg: Dict[str, Any]):
        storage = cfg["storage"]
        task = cfg["task"]
        time = cfg["time"]
        cam = cfg["cameras"]
        robot = cfg["robot"]
        teleop = cfg["teleop"]

        # global config
        self.repo_id: str = cfg["repo_id"]
        self.debug: bool = cfg.get("debug", True)
        self.fps: str = cfg.get("fps", 15)
        self.dataset_path: str = HF_LEROBOT_HOME / self.repo_id
        self.user_info: str = cfg.get("user_notes", None)

        # teleop config
        teleop_robot = teleop["robot"]
        teleop_placo = teleop["placo"]
        teleop_gripper = teleop["gripper"]
        self.scale_factor: float = teleop["scale_factor"]
        self.control_mode: str = teleop.get("control_mode", "vrteleop")
        self.R_headset_world: list[float] = teleop["R_headset_world"]
        self.left_robot_ip: str = teleop_robot["left_robot_ip"]
        self.right_robot_ip: str = teleop_robot["right_robot_ip"]
        self.visualize_placo: bool = teleop_placo["visualize_placo"]
        self.teleop_servo_time: float = teleop_placo["servo_time"]
        self.robot_urdf_path: str = teleop_placo["robot_urdf_path"]
        self.trigger_reverse: bool = teleop_gripper["trigger_reverse"]
        self.trigger_threshold: float = teleop_gripper["trigger_threshold"]
        self.close_position: float = teleop_gripper["close_position"]
        self.open_position: float = teleop_gripper["open_position"]

        # robot config
        robot_gripper = robot["gripper"]
        self.left_robot_ip: str = robot["left_robot_ip"]
        self.right_robot_ip: str = robot["right_robot_ip"]
        self.robot_servo_time: float = robot["servo_time"]
        self.gain: float = robot["gain"]
        self.lookahead_time: float = robot["lookahead_time"]
        self.use_gripper: bool = robot_gripper["use_gripper"]
        self.close_threshold: float = robot_gripper["close_threshold"]
        self.gripper_bin_threshold: float = robot_gripper["gripper_bin_threshold"]
        self.gripper_reverse: bool = robot_gripper["gripper_reverse"]
        self.left_gripper_port: int = robot_gripper["left_gripper_port"]
        self.right_gripper_port: int = robot_gripper["right_gripper_port"]

        # task config
        self.num_episodes: int = task.get("num_episodes", 1)
        self.display: bool = task.get("display", True)
        self.task_description: str = task.get("description", "default task")
        self.resume: bool = task.get("resume", "False")
        self.resume_dataset: str = task["resume_dataset"]

        # time config
        self.episode_time_sec: int = time.get("episode_time_sec", 60)
        self.reset_time_sec: int = time.get("reset_time_sec", 10)
        self.save_mera_period: int = time.get("save_mera_period", 1)

        # cameras config
        self.left_wrist_cam_serial: str = cam["left_wrist_cam_serial"]
        self.exterior_cam_serial: str = cam["exterior_cam_serial"]
        self.right_wrist_cam_serial: str = cam["right_wrist_cam_serial"]
        self.width: int = cam["width"]
        self.height: int = cam["height"]

        # storage config
        self.push_to_hub: bool = storage.get("push_to_hub", False)

def handle_incomplete_dataset(dataset_path):
    if dataset_path.exists():
        logging.info(f"====== [WARNING] Detected an incomplete dataset folder: {dataset_path} ======")
        termios.tcflush(sys.stdin, termios.TCIFLUSH)
        ans = input("Do you want to delete it? (y/n): ").strip().lower()
        if ans == "y":
            logging.info(f"====== [DELETE] Removing folder: {dataset_path} ======")
            shutil.rmtree(dataset_path, ignore_errors=True)  # Delete only this specific dataset folder
            logging.info("====== [DONE] Incomplete dataset folder deleted successfully. ======")
        else:
            logging.info("====== [KEEP] Incomplete dataset folder retained, please check manually. ======")

def listen_xrclient(xr_client: XrClient, events: Dict[str, Any], stop_signal: threading.Event):
    while not stop_signal.is_set():
        # Listen for VR input events and update the events dictionary
        if xr_client.get_button_state_by_name("A"):
            events["exit_early"] = True

        if xr_client.get_button_state_by_name("X"):
            events["rerecord_episode"] = True
            events["exit_early"] = True

        if xr_client.get_button_state_by_name("Y"):
            events["stop_recording"] = True
            events["exit_early"] = True
        if xr_client.get_button_state_by_name("left_menu_button"):
            events["keyboard_interrupt"] = True
        time.sleep(0.1)

def ensure_events_flag(events: Dict[str, Any], flag: bool = False):
    events["rerecord_episode"] = flag
    events["exit_early"] = flag

def check_keyboard_interrupt(events: Dict[str, Any]):
    if events["keyboard_interrupt"]:
        os.kill(os.getpid(), signal.SIGINT)

def run_record(record_cfg: RecordConfig):
    try:
        dataset_name, data_version = generate_dataset_name(record_cfg)

        # Create RealSenseCamera configurations
        left_wrist_image_cfg = RealSenseCameraConfig(serial_number_or_name=record_cfg.left_wrist_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.width,
                                        height=record_cfg.height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        right_wrist_image_cfg = RealSenseCameraConfig(serial_number_or_name=record_cfg.right_wrist_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.width,
                                        height=record_cfg.height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        exterior_image_cfg = RealSenseCameraConfig(serial_number_or_name=record_cfg.exterior_cam_serial,
                                        fps=record_cfg.fps,
                                        width=record_cfg.width,
                                        height=record_cfg.height,
                                        color_mode=ColorMode.RGB,
                                        use_depth=False,
                                        rotation=Cv2Rotation.NO_ROTATION)

        # Create the robot and teleoperator configurations
        camera_config = {"left_wrist_image": left_wrist_image_cfg, "right_wrist_image": right_wrist_image_cfg, "exterior_image": exterior_image_cfg}
        # Initialize XrClient
        xr_client = XrClient()
        teleop_config = VRTeleopConfig(        
            left_robot_ip=record_cfg.left_robot_ip,
            right_robot_ip=record_cfg.right_robot_ip,
            xr_client=xr_client,
            trigger_reverse=record_cfg.trigger_reverse,
            trigger_threshold=record_cfg.trigger_threshold,
            close_position=record_cfg.close_position,
            open_position=record_cfg.open_position,
            servo_time=record_cfg.teleop_servo_time,
            fps=record_cfg.fps,
            scale_factor=record_cfg.scale_factor,
            R_headset_world=record_cfg.R_headset_world,
            robot_urdf_path=record_cfg.robot_urdf_path,
            visualize_placo=record_cfg.visualize_placo,
            control_mode=record_cfg.control_mode)
        
        robot_config = URConfig(
            left_robot_ip=record_cfg.left_robot_ip,
            right_robot_ip=record_cfg.right_robot_ip,
            gain=record_cfg.gain,
            servo_time=record_cfg.robot_servo_time,
            lookahead_time=record_cfg.lookahead_time,
            left_gripper_port=record_cfg.left_gripper_port,
            right_gripper_port=record_cfg.right_gripper_port,
            cameras = camera_config,
            debug = record_cfg.debug,
            close_threshold = record_cfg.close_threshold,
            use_gripper = record_cfg.use_gripper,
            gripper_reverse = record_cfg.gripper_reverse,
            gripper_bin_threshold = record_cfg.gripper_bin_threshold
        )
        # Initialize the robot and teleoperator
        robot = UR(robot_config)
        teleop = VRTeleop(teleop_config)

        # Configure the dataset features
        action_features = hw_to_dataset_features(robot.action_features, "action")
        obs_features = hw_to_dataset_features(robot.observation_features, "observation", use_video=True)
        dataset_features = {**action_features, **obs_features}

        if record_cfg.resume:
            dataset = LeRobotDataset(
                dataset_name,
            )

            if hasattr(robot, "cameras") and len(robot.cameras) > 0:
                dataset.start_image_writer()
            sanity_check_dataset_robot_compatibility(dataset, robot, record_cfg.fps, dataset_features)
        else:
            # # Create the dataset
            dataset = LeRobotDataset.create(
                repo_id=dataset_name,
                fps=record_cfg.fps,
                features=dataset_features,
                robot_type=robot.name,
                use_videos=True,
                image_writer_threads=4,
            )
        # Set the episode metadata buffer size to 1, so that each episode is saved immediately
        dataset.meta.metadata_buffer_size = record_cfg.save_mera_period

        # Initialize the keyboard listener and rerun visualization
        _, events = init_keyboard_listener()
        init_rerun(session_name="recording")

        # Create processor
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

        robot.connect()
        teleop.connect()
        # listen the xrclient input
        stop_signal = threading.Event()
        threading.Thread(target=listen_xrclient, args=(xr_client, events, stop_signal), daemon=True).start()

        episode_idx = 0
        events["stop_recording"] = False # ensure stop_recording are reset
        events["keyboard_interrupt"] = False
        ensure_events_flag(events, False) # ensure flags are reset

        while episode_idx < record_cfg.num_episodes and not events["stop_recording"]:
            logging.info(f"====== [RECORD] Recording episode {episode_idx + 1} of {record_cfg.num_episodes} ======")
            record_loop(
                robot=robot,
                events=events,
                fps=record_cfg.fps,
                teleop=teleop,
                teleop_action_processor=teleop_action_processor,
                robot_action_processor=robot_action_processor,
                robot_observation_processor=robot_observation_processor,
                dataset=dataset,
                control_time_s=record_cfg.episode_time_sec,
                single_task=record_cfg.task_description,
                display_data=record_cfg.display,
            )

            if events["rerecord_episode"]:
                logging.info("Re-recording episode")
                events["rerecord_episode"] = False
                events["exit_early"] = False
                dataset.clear_episode_buffer()
                continue

            dataset.save_episode()

            ensure_events_flag(events, False) # ensure flags are reset

            # Reset the environment if not stopping or re-recording
            if not events["stop_recording"] and (episode_idx < record_cfg.num_episodes - 1 or events["rerecord_episode"]):
                logging.info("Please press the 'B' button on the VR controller to `reset` the environment.")
                while not xr_client.get_button_state_by_name("B"):
                    check_keyboard_interrupt(events)
                    time.sleep(0.1) 

                logging.info("====== [RESET] Resetting the environment ======")
                record_loop(
                    robot=robot,
                    events=events,
                    fps=record_cfg.fps,
                    teleop=teleop,
                    teleop_action_processor=teleop_action_processor,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    control_time_s=record_cfg.reset_time_sec,
                    single_task=record_cfg.task_description,
                    display_data=record_cfg.display,
                )

                logging.info("Please press the 'B' button on the VR controller to start the `next episode`.")
                while not xr_client.get_button_state_by_name("B"):
                    check_keyboard_interrupt(events)
                    time.sleep(0.1)

            ensure_events_flag(events, False) # ensure flags are reset
            episode_idx += 1

        # Clean up
        logging.info("Stop recording")
        robot.disconnect()
        teleop.disconnect()
        stop_signal.set()
        dataset.finalize()

        update_dataset_info(record_cfg, dataset_name, data_version)
        if record_cfg.push_to_hub:
            dataset.push_to_hub()

    except Exception as e:
        logging.info(f"====== [ERROR] {e} ======")
        dataset_path = Path(HF_LEROBOT_HOME) / dataset_name
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)

    except KeyboardInterrupt:
        logging.info("\n====== [INFO] Ctrl+C detected, cleaning up incomplete dataset... ======")
        dataset_path = Path(HF_LEROBOT_HOME) / dataset_name
        handle_incomplete_dataset(dataset_path)
        sys.exit(1)


def main():
    parent_path = Path(__file__).resolve().parent
    cfg_path = parent_path.parent / "config" / "cfg.yaml"
    with open(cfg_path, 'r') as f:
        cfg = yaml.safe_load(f)

    record_cfg = RecordConfig(cfg["record"])
    run_record(record_cfg)