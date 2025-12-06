def main():
    print("""
==================================================
 VR Teleoperation Utilities - Command Reference
==================================================

Core Commands:
  vr-record           Record teleoperation dataset
  vr-replay           Replay a recorded dataset
  vr-visualize        Visualize recorded dataset

Tool Commands:
  tools-check-dataset   Check local dataset information
  tools-check-rs        Retrieve connected RealSense camera serial numbers

Shell Tools:
  map_gripper.sh        Map Gripper Serial Port
  check_master_port.sh  Get the Master Arm's Persistent Serial Identifier

Test Commands:
  test-gripper-ctrl     Run gripper control command (operate the gripper)

--------------------------------------------------
 Tip: Use 'vr-help' anytime to see this summary.
==================================================
""")
