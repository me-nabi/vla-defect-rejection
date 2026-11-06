"""
Scripted demonstration collection for the conveyor belt defect rejection task.
Records pick-and-place trajectories in HDF5 format for SmolVLA fine-tuning.
"""
import os
import h5py
import numpy as np
from PIL import Image
from libero.libero.envs import TASK_MAPPING

BDDL_FILE = "/home/nabi/LIBERO/libero/libero/bddl_files/defect_rejection/pick_defective_bottle_from_belt.bddl"
OUTPUT_DIR = "/home/nabi/LIBERO/demos"
NUM_DEMOS = 50
MAX_STEPS = 300
IMG_SIZE = 256

CONTROLLER_CONFIG = {
    "type": "OSC_POSE",
    "input_max": 1, "input_min": -1,
    "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
    "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
    "kp": 150, "damping": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300], "damping_limits": [0, 10],
    "position_limits": None, "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}


def compute_position_action(current_pos, target_pos, gain=10.0, max_action=1.0):
    delta = target_pos - current_pos
    action = np.clip(delta * gain, -max_action, max_action)
    return action


def run_scripted_policy(env, obs):
    """Execute a scripted pick-and-place. Returns (trajectory, success)."""
    trajectory = []

    bottle_pos = obs["glass_bottle_1_pos"].copy()
    bin_pos = obs["reject_bin_1_pos"].copy()

    grasp_height_offset = 0.05
    approach_height = bottle_pos[2] + 0.17
    grasp_z = bottle_pos[2] + grasp_height_offset
    lift_height = bottle_pos[2] + 0.22
    drop_height = bin_pos[2] + 0.10

    noise_xy = np.random.uniform(-0.006, 0.006, size=2)
    noise_approach = np.random.uniform(-0.01, 0.01)
    noise_gain = np.random.uniform(10.0, 14.0)

    # Gripper convention: -1 = open, +1 = close
    phases = [
        ("approach_above", lambda: np.array([bottle_pos[0] + noise_xy[0],
                                              bottle_pos[1] + noise_xy[1],
                                              approach_height + noise_approach]), -1.0),
        ("descend",        lambda: np.array([bottle_pos[0], bottle_pos[1], grasp_z]), -1.0),
        ("grasp",          None, 1.0),
        ("lift",           lambda: np.array([obs["robot0_eef_pos"][0],
                                              obs["robot0_eef_pos"][1],
                                              lift_height]), 1.0),
        ("move_to_bin",    lambda: np.array([bin_pos[0], bin_pos[1], drop_height]), 1.0),
        ("release",        None, -1.0),
        ("retreat",        lambda: np.array([bin_pos[0], bin_pos[1],
                                              drop_height + 0.10]), -1.0),
    ]

    step_count = 0
    for phase_name, target_fn, gripper_cmd in phases:
        if target_fn is None:
            for _ in range(15):
                if step_count >= MAX_STEPS:
                    break
                action = np.zeros(7)
                action[6] = gripper_cmd
                record = _record_step(obs, action)
                trajectory.append(record)
                obs, _, done, _ = env.step(action)
                step_count += 1

                if phase_name == "descend":
                    bottle_pos = obs["glass_bottle_1_pos"].copy()
        else:
            for _ in range(80):
                if step_count >= MAX_STEPS:
                    break
                target = target_fn()
                eef_pos = obs["robot0_eef_pos"]
                pos_action = compute_position_action(eef_pos, target, gain=noise_gain)
                action = np.zeros(7)
                action[:3] = pos_action
                action[6] = gripper_cmd

                record = _record_step(obs, action)
                trajectory.append(record)
                obs, _, done, _ = env.step(action)
                step_count += 1

                if phase_name in ("approach_above", "descend"):
                    bottle_pos = obs["glass_bottle_1_pos"].copy()

                dist = np.linalg.norm(eef_pos - target)
                if dist < 0.01:
                    break

    success = env._check_success()
    return trajectory, success, obs


def _record_step(obs, action):
    return {
        "agentview_image": obs["agentview_image"][::-1].copy(),
        "eye_in_hand_image": obs["robot0_eye_in_hand_image"][::-1].copy(),
        "eef_pos": obs["robot0_eef_pos"].copy(),
        "eef_quat": obs["robot0_eef_quat"].copy(),
        "gripper_qpos": obs["robot0_gripper_qpos"].copy(),
        "joint_pos": obs["robot0_joint_pos"].copy(),
        "action": action.copy(),
        "bottle_pos": obs["glass_bottle_1_pos"].copy(),
        "bin_pos": obs["reject_bin_1_pos"].copy(),
    }


def save_demos_hdf5(all_demos, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with h5py.File(output_path, "w") as f:
        f.attrs["num_demos"] = len(all_demos)
        f.attrs["env_name"] = "conveyor_belt_manipulation"

        for i, demo in enumerate(all_demos):
            grp = f.create_group(f"demo_{i}")
            grp.attrs["num_steps"] = len(demo)

            agentview_imgs = np.stack([s["agentview_image"] for s in demo])
            hand_imgs = np.stack([s["eye_in_hand_image"] for s in demo])
            actions = np.stack([s["action"] for s in demo])
            eef_pos = np.stack([s["eef_pos"] for s in demo])
            eef_quat = np.stack([s["eef_quat"] for s in demo])
            gripper_qpos = np.stack([s["gripper_qpos"] for s in demo])
            joint_pos = np.stack([s["joint_pos"] for s in demo])

            state = np.concatenate([eef_pos, eef_quat, gripper_qpos], axis=1)

            obs_grp = grp.create_group("obs")
            obs_grp.create_dataset("agentview_image", data=agentview_imgs, compression="gzip", compression_opts=4)
            obs_grp.create_dataset("eye_in_hand_image", data=hand_imgs, compression="gzip", compression_opts=4)
            obs_grp.create_dataset("eef_pos", data=eef_pos)
            obs_grp.create_dataset("eef_quat", data=eef_quat)
            obs_grp.create_dataset("gripper_qpos", data=gripper_qpos)
            obs_grp.create_dataset("joint_pos", data=joint_pos)
            obs_grp.create_dataset("state", data=state)

            grp.create_dataset("actions", data=actions)

    print(f"Saved {len(all_demos)} demos to {output_path}")


def main():
    env = TASK_MAPPING["conveyor_belt_manipulation"](
        bddl_file_name=BDDL_FILE,
        robots=["Panda"],
        controller_configs=CONTROLLER_CONFIG,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        camera_heights=IMG_SIZE,
        camera_widths=IMG_SIZE,
        camera_names=["agentview", "robot0_eye_in_hand"],
        reward_shaping=True,
        control_freq=20,
        belt_active=False,
    )

    successful_demos = []
    total_attempts = 0

    print(f"Collecting {NUM_DEMOS} demonstrations...")

    while len(successful_demos) < NUM_DEMOS:
        total_attempts += 1
        obs = env.reset()
        trajectory, success, final_obs = run_scripted_policy(env, obs)

        if success:
            successful_demos.append(trajectory)
            print(f"  Demo {len(successful_demos)}/{NUM_DEMOS} — {len(trajectory)} steps (attempt {total_attempts})")
        else:
            bottle_pos = final_obs["glass_bottle_1_pos"]
            bin_pos = final_obs["reject_bin_1_pos"]
            dist = np.linalg.norm(bottle_pos[:2] - bin_pos[:2])
            print(f"  Attempt {total_attempts} FAILED — bottle-bin dist: {dist:.3f}")

        if total_attempts > NUM_DEMOS * 3:
            print(f"Stopping after {total_attempts} attempts. Got {len(successful_demos)} successful demos.")
            break

    env.close()

    if successful_demos:
        output_path = os.path.join(OUTPUT_DIR, "conveyor_demos.hdf5")
        save_demos_hdf5(successful_demos, output_path)

        sample_dir = os.path.join(OUTPUT_DIR, "sample_frames")
        os.makedirs(sample_dir, exist_ok=True)
        demo = successful_demos[0]
        for idx in [0, len(demo)//4, len(demo)//2, 3*len(demo)//4, len(demo)-1]:
            img = demo[idx]["agentview_image"]
            Image.fromarray(img).save(os.path.join(sample_dir, f"demo0_step{idx:03d}.png"))
        print(f"Saved sample frames to {sample_dir}")

    success_rate = len(successful_demos) / total_attempts * 100 if total_attempts > 0 else 0
    print(f"\nDone: {len(successful_demos)}/{total_attempts} successful ({success_rate:.0f}%)")


if __name__ == "__main__":
    main()
