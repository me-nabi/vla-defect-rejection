"""Convert HDF5 demos to LeRobot dataset format for SmolVLA fine-tuning."""
import sys
sys.path.insert(0, "/home/nabi/LIBERO")

import h5py
import numpy as np
from PIL import Image
from lerobot.datasets.lerobot_dataset import LeRobotDataset

HDF5_PATH = "/home/nabi/LIBERO/demos/conveyor_demos.hdf5"
REPO_ID = "conveyor-defect-rejection"
DATASET_ROOT = "/home/nabi/LIBERO/lerobot_dataset"
TASK_DESCRIPTION = "pick the defective bottle from the conveyor belt and place it in the reject bin"
FPS = 20

FEATURES = {
    "observation.images.image": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.images.image2": {
        "dtype": "image",
        "shape": (256, 256, 3),
        "names": ["height", "width", "channels"],
    },
    "observation.state": {
        "dtype": "float32",
        "shape": (8,),
        "names": {
            "motors": [
                "joint_0", "joint_1", "joint_2", "joint_3",
                "joint_4", "joint_5", "joint_6", "gripper",
            ]
        },
    },
    "action": {
        "dtype": "float32",
        "shape": (7,),
        "names": {
            "motors": [
                "x", "y", "z", "rx", "ry", "rz", "gripper",
            ]
        },
    },
}


def main():
    print(f"Loading HDF5 demos from {HDF5_PATH}")
    h5 = h5py.File(HDF5_PATH, "r")
    num_demos = h5.attrs["num_demos"]
    print(f"Found {num_demos} demos")

    print(f"Creating LeRobot dataset at {DATASET_ROOT}")
    dataset = LeRobotDataset.create(
        repo_id=REPO_ID,
        fps=FPS,
        features=FEATURES,
        root=DATASET_ROOT,
        robot_type="panda",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=4,
    )

    total_frames = 0
    for ep_idx in range(num_demos):
        demo = h5[f"demo_{ep_idx}"]
        num_steps = demo.attrs["num_steps"]

        agentview = demo["obs"]["agentview_image"][:]
        hand_view = demo["obs"]["eye_in_hand_image"][:]
        joint_pos = demo["obs"]["joint_pos"][:]
        gripper_qpos = demo["obs"]["gripper_qpos"][:]
        actions = demo["actions"][:]

        # Build 8D state: joint_pos(7) + gripper_width(1)
        gripper_width = gripper_qpos[:, 0:1]
        state = np.concatenate([joint_pos, gripper_width], axis=1).astype(np.float32)

        for step_idx in range(num_steps):
            frame = {
                "observation.images.image": Image.fromarray(agentview[step_idx]),
                "observation.images.image2": Image.fromarray(hand_view[step_idx]),
                "observation.state": state[step_idx],
                "action": actions[step_idx].astype(np.float32),
                "task": TASK_DESCRIPTION,
            }
            dataset.add_frame(frame)

        dataset.save_episode()
        total_frames += num_steps

        if (ep_idx + 1) % 10 == 0:
            print(f"  Converted {ep_idx + 1}/{num_demos} episodes ({total_frames} frames)")

    print("Finalizing dataset...")
    dataset.finalize()

    h5.close()
    print(f"\nDone! {num_demos} episodes, {total_frames} frames")
    print(f"Dataset saved to: {DATASET_ROOT}/{REPO_ID}")


if __name__ == "__main__":
    main()
