"""Run scripted policy on the industrial conveyor belt and record video."""
import sys
sys.path.insert(0, "/home/nabi/LIBERO")

import os
import numpy as np
import imageio
from PIL import Image, ImageDraw
from libero.libero.envs import TASK_MAPPING

BDDL_FILE = "/home/nabi/LIBERO/libero/libero/bddl_files/defect_rejection/pick_defective_bottle_from_belt.bddl"
VIDEO_PATH = "/home/nabi/LIBERO/rollout_video.mp4"
FRAMES_DIR = "/home/nabi/LIBERO/rollout_frames"
MAX_STEPS = 400

CONTROLLER_CONFIG = {
    "type": "OSC_POSE",
    "input_max": 1, "input_min": -1,
    "output_max": [0.12, 0.12, 0.12, 0.5, 0.5, 0.5],
    "output_min": [-0.12, -0.12, -0.12, -0.5, -0.5, -0.5],
    "kp": 180, "damping": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300], "damping_limits": [0, 10],
    "position_limits": None, "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}


def compute_action(current_pos, target_pos, gain=10.0):
    delta = target_pos - current_pos
    return np.clip(delta * gain, -1.0, 1.0)


def add_label(frame, text):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    draw.rectangle([0, 0, img.width, 22], fill=(20, 20, 20))
    draw.text((8, 4), text, fill=(220, 220, 220))
    return np.array(img)


def record_frame(obs):
    agentview = obs["agentview_image"][::-1].copy()
    hand = obs["robot0_eye_in_hand_image"][::-1].copy()
    h, w = agentview.shape[:2]
    combined = np.zeros((h, w * 2, 3), dtype=np.uint8)
    combined[:, :w] = agentview
    combined[:, w:] = hand
    return combined


def make_env(belt_active=True, belt_speed=0.002):
    return TASK_MAPPING["conveyor_belt_manipulation"](
        bddl_file_name=BDDL_FILE,
        robots=["Panda"],
        controller_configs=CONTROLLER_CONFIG,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        camera_heights=256,
        camera_widths=256,
        camera_names=["agentview", "robot0_eye_in_hand"],
        reward_shaping=True,
        control_freq=20,
        belt_active=belt_active,
        belt_speed=belt_speed,
    )


def run_scripted_episode(env, obs):
    """Scripted pick-and-place that tracks the moving bottle.

    Belt stays active during approach (bottle moves visually),
    then pauses during grasp/lift/place for reliable manipulation.
    """
    frames = []
    step = 0
    grasp_count = 0
    release_count = 0

    phase = "approach"

    bottle_pos = obs["glass_bottle_1_pos"].copy()
    bin_pos = obs["reject_bin_1_pos"].copy()
    eef_pos = obs["robot0_eef_pos"].copy()
    print(f"  Init: bottle={bottle_pos}, bin={bin_pos}, eef={eef_pos}")

    while step < MAX_STEPS and phase != "done":
        bottle_pos = obs["glass_bottle_1_pos"].copy()
        bin_pos = obs["reject_bin_1_pos"].copy()
        eef_pos = obs["robot0_eef_pos"].copy()

        action = np.zeros(7)

        if phase == "approach":
            target = np.array([bottle_pos[0], bottle_pos[1], bottle_pos[2] + 0.12])
            action[:3] = compute_action(eef_pos, target, gain=12.0)
            action[6] = -1.0
            dist_xy = np.linalg.norm(eef_pos[:2] - bottle_pos[:2])
            if dist_xy < 0.02 and abs(eef_pos[2] - target[2]) < 0.025:
                phase = "descend"
                env.belt_active = False
                print(f"  Step {step}: -> descend (belt paused)")

        elif phase == "descend":
            target = np.array([bottle_pos[0], bottle_pos[1], bottle_pos[2] + 0.045])
            action[:3] = compute_action(eef_pos, target, gain=10.0)
            action[6] = -1.0
            if eef_pos[2] < bottle_pos[2] + 0.08:
                phase = "grasp"
                print(f"  Step {step}: -> grasp (eef_z={eef_pos[2]:.4f})")

        elif phase == "grasp":
            target = np.array([bottle_pos[0], bottle_pos[1], bottle_pos[2] + 0.045])
            action[:3] = compute_action(eef_pos, target, gain=4.0)
            action[6] = 1.0
            grasp_count += 1
            if grasp_count >= 40:
                phase = "lift"
                gripper_q = obs['robot0_gripper_qpos']
                print(f"  Step {step}: -> lift (gripper_qpos={gripper_q})")

        elif phase == "lift":
            target = np.array([eef_pos[0], eef_pos[1], 1.15])
            action[:3] = compute_action(eef_pos, target, gain=10.0)
            action[6] = 1.0
            if eef_pos[2] > 1.12:
                bz = bottle_pos[2]
                phase = "to_bin"
                print(f"  Step {step}: -> to_bin (bottle_z={bz:.4f}, lifted={'YES' if bz > 0.95 else 'NO'})")

        elif phase == "to_bin":
            target = np.array([bin_pos[0], bin_pos[1], 1.10])
            action[:3] = compute_action(eef_pos, target, gain=10.0)
            action[6] = 1.0
            dist_xy = np.linalg.norm(eef_pos[:2] - bin_pos[:2])
            if dist_xy < 0.03:
                phase = "release"
                print(f"  Step {step}: -> release (above bin)")

        elif phase == "release":
            action[:3] = np.zeros(3)
            action[6] = -1.0
            release_count += 1
            if release_count >= 25:
                phase = "retreat"
                print(f"  Step {step}: -> retreat")

        elif phase == "retreat":
            target = np.array([eef_pos[0], eef_pos[1], 1.25])
            action[:3] = compute_action(eef_pos, target, gain=8.0)
            action[6] = -1.0
            if eef_pos[2] > 1.22:
                phase = "done"
                print(f"  Step {step}: -> done")

        label = f"DEFECT REJECTION | {phase.upper()} | step {step}"
        frames.append(add_label(record_frame(obs), label))
        obs, _, _, _ = env.step(action)
        step += 1

    for _ in range(min(30, MAX_STEPS - step)):
        frames.append(add_label(record_frame(obs), f"DEFECT REJECTION | COMPLETE | step {step}"))
        obs, _, _, _ = env.step(np.zeros(7))
        step += 1

    success = env._check_success()
    bottle_final = obs["glass_bottle_1_pos"].copy()
    print(f"  Final: bottle_z={bottle_final[2]:.4f}, success={success}")
    return frames, success, obs


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)

    best_frames = None

    for attempt in range(5):
        print(f"\nAttempt {attempt+1}...")

        env = make_env(belt_active=True, belt_speed=0.002)
        obs = env.reset()
        frames, success, final_obs = run_scripted_episode(env, obs)
        env.close()

        print(f"  Result: {'SUCCESS' if success else 'FAIL'}")

        if success:
            best_frames = frames
            break
        elif best_frames is None:
            best_frames = frames

    all_frames = best_frames

    indices = [0, len(all_frames)//5, 2*len(all_frames)//5, 3*len(all_frames)//5, 4*len(all_frames)//5, len(all_frames)-1]
    for i, idx in enumerate(indices):
        if idx < len(all_frames):
            Image.fromarray(all_frames[idx]).save(f"{FRAMES_DIR}/frame_{i}_{idx:03d}.png")

    print(f"\nSaving video ({len(all_frames)} frames)...")
    writer = imageio.get_writer(VIDEO_PATH, fps=20, quality=8)
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()
    print(f"Video saved: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
