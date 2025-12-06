from setuptools import setup, find_packages
from pathlib import Path
# ====== Project root ======
ROOT = Path(__file__).parent.resolve()
setup(
    name="lerobot_ur_dual_vrteleop",
    version="0.1.0",
    description="Dual UR teleoperation and dataset collection utilities with VR support",
    python_requires=">=3.10",
    packages=find_packages(where=".", include=["scripts", "scripts.*"]),
    include_package_data=True,
    scripts=[
        "scripts/tools/map_gripper.sh",
    ],
    entry_points={
        "console_scripts": [
            # core commands
            "vr-record = scripts.core.run_record:main",
            "vr-replay = scripts.core.run_replay:main",
            "vr-visualize = scripts.core.run_visualize:main",

            # tools commands (helper tools)
            "tools-check-dataset = scripts.tools.check_dataset_info:main",
            "tools-check-rs = scripts.tools.rs_devices:main",

            # test commands (testing scripts)
            "test-gripper-ctrl = scripts.test.gripper_ctrl:main",
            "test-ur-freedrive = scripts.test.ur_freedrive:main",

            # unified help command
            "vr-help = scripts.help.help_info:main",
        ]
    },
)
