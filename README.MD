# UR Dual-Arm VR Teleoperation Data Collection
<p align="center">
  <img src="assets/robot.png" alt="Record" width="300">
  <br>
</p>

## 1. Environment Setup
### 1.1 Create and Activate Conda Environment
```bash
conda create -n ur_data python=3.10
conda activate ur_data
```

### 1.2 Install lerobot
```bash
# Install specific version 0.3.4
git clone https://github.com/huggingface/lerobot.git
cd lerobot
git checkout da5d2f3e9187fa4690e6667fe8b294cae49016d6
pip install -e .
```
### 1.3 Install [XRoboToolkit](https://github.com/XR-Robotics)
- Please complete `Step 1` and `Step 3` of the `Get Started` in [XRoboToolkit](https://github.com/XR-Robotics)
- Then complete the following installation:
```bash
git clone https://github.com/XR-Robotics/XRoboToolkit-Teleop-Sample-Python.git
cd XRoboToolkit-Teleop-Sample-Python
bash setup_conda.sh --install
```
- Finally, please complete the first three steps of `Step 4` in `Get Started` to ensure the VR device can communicate with the computer.
  
### 1.4 Clone and Install VR Teleoperation Control Source
```bash
mkdir ur_data_collection && cd ur_data_collection
git clone https://github.com/scy-v/lerobot_ur_dual_vrteleop.git
cd lerobot_ur_dual_vrteleop
pip install -r requirements.txt # install all packages in editable mode
```

### 1.4 View Available Commands
Run the following command to display all available scrtips:
```bash
vr-help
# ==================================================
#  VR Teleoperation Utilities - Command Reference
# ==================================================

# Core Commands:
#   vr-record           Record teleoperation dataset
#   vr-replay           Replay a recorded dataset
#   vr-visualize        Visualize recorded dataset

# Tool Commands:
#   tools-check-dataset   Check local dataset information
#   tools-check-rs        Retrieve connected RealSense camera serial numbers

# Shell Tools:
#   map_gripper.sh        Map Gripper Serial Port

# Test Commands:
#   test-gripper-ctrl     Run gripper control command (operate the gripper)

# --------------------------------------------------
#  Tip: Use 'vr-help' anytime to see this summary.
# ==================================================
```

## 2. Obtain and Configure Necessary Parameters

### 2.1 Get RealSense Camera Serial Number
Please make sure only one camera is connected at a time.
```bash
tools-check-rs
```

### 2.2 Map Gripper Serial Port
For example, map the gripper to `/dev/ur_left_gripper` and make sure only one gripper USB device is connected:
```bash
map_gripper.sh ur_left_gripper
```
Next, enter the mapped device into the `left/right_gripper_port` field in `cfg.yaml`.

## 3. Dataset Recording

### 3.1 Upload Data to Hugging Face (Optional)
1. Set the following in `cfg.yaml`:
```yaml
push_to_hub: True
```
2. Obtain your Hugging Face account token and login:
```bash
huggingface-cli login --token ${HUGGINGFACE_TOKEN} 
huggingface-cli whoami  # Verify successful login
```

### 3.2 Start Recording
1. Turn on **Remote Control** on the UR5e teach pendant.  
2. Verify that the parameters in `cfg.yaml` are correct.  
3. To ensure standardized data collection, please read `9. Dataset Naming and Recording Details`.
4. Make sure VR device is connected to the computer, then run the following commands:
```bash
vr-record
```

<!-- <p align="center">
  <img src="assets/record.png" alt="Record" width="600">
  <br>
  <b>Figure 1: Record</b>
</p> -->

## 4. Dataset Replay
```bash
vr-replay # Make sure cfg.yaml is properly configured
```

## 5. Dataset Visualization
```bash
vr-visualize # Make sure cfg.yaml is properly configured
```
<!-- <p align="center">
  <img src="assets/visualize.png" alt="Visualization" width="600">
  <br>
  <b>Figure 2: Visualization</b>
</p> -->

## 6. Dataset Appending and Resume
If you have already recorded a dataset under the specified `repo_id`, set `resume: True` in `cfg.yaml` and enter the dataset name in `resume_dataset` to continue recording new data on the existing dataset. Then, run the following command:

```bash
vr-record
```

## 7. Merge Datasets
If you recorded data in multiple stages, ensure each dataset has a unique `repo_id`. Use the following command to merge them into a single dataset:
```bash
lerobot-edit-dataset 
    --repo_id <merged_repo_id> 
    --operation.type merge 
    --operation.repo_ids "['<repo_id_1>', '<repo_id2>']"
```
- For more dataset processing commands, see [LeRobot](https://huggingface.co/docs/lerobot/using_dataset_tools).

## 8. VR Controller Button Description

1. **Right Controller**
   1. **A Button**: Ends the recording of the current episode.
   2. **B Button**: Resets the environment.
   3. **Trigger**: Press to control the robot; release to stop control.
   4. **Grip Button**: Press to close the gripper; release to open it.

2. **Left Controller**
   1. **X Button**: Restart the recording of the current episode.
   2. **Y Button**: End recording, save data, and exit.
   3. **Menu Button/Menu+A Button**: Throws an exception and terminates the program, with an option to delete the current recorded data (Y/N).
   4. **Trigger**: Press to control the robot; release to stop control.
   5. **Grip Button**: Press to close the gripper; release to open it.

3. **Recording Procedures**
   1. `Start recording (Trigger and Grip can be used freely)` -> `Press A to end the current episode` -> `Press B to reset the robot (Trigger and Grip allowed)` -> `Press A to finish reset` -> `Press B to start the next episode` -> `Repeat until Y is pressed to exit normally`
   2. `Start recording (Trigger and Grip can be used freely)` -> `Press Menu to throw an exception and choose whether to delete the recorded data (Y/N), then exit`
   3. `Start recording (Trigger and Grip can be used freely)` -> `Press Y to finish recording and save data`

  
## 9. Dataset Naming and Recording Details
### 1. Dataset Naming
<p align="center">
  <img src="assets/dataset.png" alt="dataset" width="600">
  <br>
  <b>Figure 3: Dataset</b>
</p>

<p align="center">
  <img src="assets/dataset_info.png" alt="dataset_info" width="600">
  <br>
  <b>Figure 4: Dataset Info</b>
</p>

1. The dataset is by default stored in the `~/.cache/huggingface/lerobot` directory and contains three types of items:

   - `dataset_info.txt`: Automatically records local dataset information, including the following fields: `record_id`, `name`, `task`, `date`, `version`, `user_info`, and `type`. The `user_info` field can be annotated via the `user_notes` entry in `cfg.yaml`.

   - `dataset_info_backup`: When `tools-check-dataset` is used to manually update `dataset_info.txt`, this folder stores backups of the previous records.

   - Dataset folders: Stores the actual dataset contents.

2. Dataset names follow the format `[description]_[date]_[version]`, where:
   - `description` comes from `cfg.yaml` as `repo_id=<user_name>/<description>`;
   - `date` is generated automatically;
   - `version` is incremented automatically if a dataset with the same `repo_id` already exists.
3. Description naming rule: `task.description -> Verb_SourceObject_prep_TargetObject`.  
   For example:  
   `Pick up the green cube and put it into the trash bin -> pick_greencube_into_trashbin`.

### 2. Recording Instructions
1. Complete step `2. Get and configure required parameters`.
2. Fill in the task instruction in `cfg.yaml` under `task.description` and set `repo_id` according to the dataset naming rules.
3. Check and adjust other parameters in `cfg.yaml` to ensure correctness.
4. Run `vr-record`, and then follow `8. VR Controller Button Description` to complete the recording.
5. After recording, a `dataset_info.txt` file will be automatically created in the same directory to store local dataset information. If you have manually deleted any datasets, update the `dataset_info.txt` using:
```bash
tools-check-dataset
```