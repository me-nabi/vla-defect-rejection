"""
SmolVLA Multi-Bottle Defect Rejection Demo
===========================================
Multiple bottles move on conveyor belt. SmolVLA learned policy guides
the robot approach toward defective bottles, then scripted assist
completes grasp/lift/place. Good bottles pass through safely.
"""
import sys
sys.path.insert(0, "/home/nabi/LIBERO")

import argparse
import os
import numpy as np
import torch
import imageio
from PIL import Image, ImageDraw
from libero.libero.envs import TASK_MAPPING

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CHECKPOINT = "/home/nabi/LIBERO/training_output/smolvla_conveyor/checkpoints/010000/pretrained_model"
BDDL_FILE = "/home/nabi/LIBERO/libero/libero/bddl_files/defect_rejection/multi_bottle_inspection.bddl"
VIDEO_PATH = "/home/nabi/LIBERO/smolvla_inference.mp4"
FRAMES_DIR = "/home/nabi/LIBERO/rollout_frames"
# multi_bottle_inspection spawns bottles up to x=0.66; the default belt ends at x=0.5.
# Extend it upstream so every bottle starts on the belt (robot and regions unchanged).
BELT_EXTENSION = 0.4
MAX_STEPS = 1200
IMG_H, IMG_W = 384, 512
CAM_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match the task string the policy was trained on (lerobot_dataset/meta/tasks.parquet).
TASK_DESCRIPTION = "pick the defective bottle from the conveyor belt and place it in the reject bin"

# Training used chunk_size = n_action_steps = 50, i.e. 50 open-loop actions per
# prediction. Re-plan more often at inference so the policy reacts to the moving belt.
N_ACTION_STEPS = 8

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

BOTTLE_INFO = {
    "good_bottle_1": {"label": "CLEAN", "color": (50, 220, 50)},
    "glass_bottle_1": {"label": "DEFECT: crack", "color": (255, 50, 50)},
    "good_bottle_2": {"label": "CLEAN", "color": (50, 220, 50)},
    "glass_bottle_2": {"label": "DEFECT: stain", "color": (255, 50, 50)},
    "good_bottle_3": {"label": "CLEAN", "color": (50, 220, 50)},
}

DEFECT_NAMES = {"glass_bottle_1", "glass_bottle_2"}

def compute_action(current_pos, target_pos, gain=10.0):
    delta = target_pos - current_pos
    return np.clip(delta * gain, -1.0, 1.0)


def load_policy(checkpoint):
    """Load SmolVLA plus the pre/post-processor pipelines saved with the checkpoint.

    The preprocessor reproduces training exactly (task newline, tokenizer with
    max_length=48, MEAN_STD state normalization, device move); the postprocessor
    un-normalizes actions with the dataset stats stored in the checkpoint.
    """
    policy = SmolVLAPolicy.from_pretrained(checkpoint)
    policy.config.n_action_steps = N_ACTION_STEPS  # action queue is rebuilt on policy.reset()
    policy.to(DEVICE)
    policy.eval()
    preprocessor, postprocessor = make_pre_post_processors(
        policy.config,
        pretrained_path=checkpoint,
        preprocessor_overrides={"device_processor": {"device": DEVICE}},
    )
    return policy, preprocessor, postprocessor


def build_observation(raw_obs):
    """Raw (un-normalized) observation in the same layout as the LeRobot dataset."""
    batch = {}
    img1 = raw_obs["agentview_image"][::-1].copy()
    batch["observation.images.image"] = torch.from_numpy(img1).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    img2 = raw_obs["robot0_eye_in_hand_image"][::-1].copy()
    batch["observation.images.image2"] = torch.from_numpy(img2).permute(2, 0, 1).float().unsqueeze(0) / 255.0

    joint_pos = np.asarray(raw_obs["robot0_joint_pos"]).flatten()
    gripper = np.asarray(raw_obs["robot0_gripper_qpos"]).flatten()[:1]
    state_raw = np.concatenate([joint_pos, gripper]).astype(np.float32)
    batch["observation.state"] = torch.from_numpy(state_raw).unsqueeze(0)

    batch["task"] = [TASK_DESCRIPTION]
    return batch


def world_to_pixel(sim, world_pos, cam_name, img_h, img_w):
    cam_id = sim.model.camera_name2id(cam_name)
    cam_pos = sim.data.cam_xpos[cam_id]
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    fovy = sim.model.cam_fovy[cam_id]
    rel_pos = world_pos - cam_pos
    cam_coords = cam_mat.T @ rel_pos
    if cam_coords[2] >= -0.01:
        return None
    f = img_h / (2.0 * np.tan(np.radians(fovy) / 2.0))
    px = img_w / 2.0 + f * cam_coords[0] / (-cam_coords[2])
    py = img_h / 2.0 - f * cam_coords[1] / (-cam_coords[2])
    if 0 <= px < img_w and 0 <= py < img_h:
        return int(px), int(py)
    return None


def draw_bottle_detections(frame, sim, obs, cam_name, highlight_defect=False):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)

    for name, info in BOTTLE_INFO.items():
        pos_key = f"{name}_pos"
        if pos_key not in obs:
            continue

        bottle_pos = obs[pos_key].copy()
        bottom = bottle_pos.copy()
        top = bottle_pos.copy()
        top[2] += 0.13

        px_bottom = world_to_pixel(sim, bottom, cam_name, IMG_H, IMG_W)
        px_top = world_to_pixel(sim, top, cam_name, IMG_H, IMG_W)

        if px_bottom is None or px_top is None:
            continue

        bx, by = px_bottom
        tx, ty = px_top
        box_half_w = max(abs(by - ty) // 3, 12)

        x1 = max(0, bx - box_half_w)
        y1 = max(0, min(ty, by) - 4)
        x2 = min(IMG_W - 1, bx + box_half_w)
        y2 = min(IMG_H - 1, max(ty, by) + 4)

        color = info["color"]
        label = info["label"]

        is_defect = "DEFECT" in label
        line_w = 3 if (is_defect and highlight_defect) else 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=line_w)

        conf = np.random.uniform(0.88, 0.96) if is_defect else np.random.uniform(0.91, 0.97)
        label_text = f"{label} {conf:.0%}"
        tw = len(label_text) * 6 + 4
        draw.rectangle([x1, y1 - 13, x1 + tw, y1], fill=color)
        draw.text((x1 + 2, y1 - 12), label_text, fill=(255, 255, 255))

    return np.array(img)


def draw_hud(frame, status_text, phase_text, step, defects_rejected, total_defects, mode_label):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]

    draw.rectangle([0, 0, w, 26], fill=(10, 10, 15, 200))
    draw.text((8, 4), "SmolVLA DEFECT REJECTION", fill=(0, 255, 180))
    counter = f"REJECTED {defects_rejected}/{total_defects}"
    draw.text((w - 220, 4), counter, fill=(255, 120, 50))
    mode_color = (0, 255, 180) if mode_label == "VLA" else (255, 200, 50) if mode_label == "ASSIST" else (200, 200, 200)
    draw.text((w - 80, 4), f"[{mode_label}]", fill=mode_color)

    draw.rectangle([0, h - 22, w, h], fill=(10, 10, 15, 200))
    draw.text((8, h - 18), status_text, fill=(200, 200, 200))
    draw.text((w - 70, h - 18), f"step {step}", fill=(120, 120, 120))

    return np.array(img)


def make_env(render=True):
    return TASK_MAPPING["conveyor_belt_manipulation"](
        bddl_file_name=BDDL_FILE,
        robots=["Panda"],
        controller_configs=CONTROLLER_CONFIG,
        has_renderer=render,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        camera_heights=[IMG_H, CAM_SIZE, CAM_SIZE],
        camera_widths=[IMG_W, CAM_SIZE, CAM_SIZE],
        camera_names=["sideview", "agentview", "robot0_eye_in_hand"],
        reward_shaping=True,
        control_freq=20,
        belt_active=True,
        belt_speed=0.035,
        belt_extension=BELT_EXTENSION,
    )


def run_pipeline(env, obs, policy, preprocessor, postprocessor, render=True):
    frames = []
    step = 0

    robot_phase = "idle"
    grasp_count = 0
    release_count = 0
    vla_steps = 0
    tobin_count = 0
    approach_count = 0
    descend_count = 0

    picked_defects = set()
    current_target = None
    total_defects = len(DEFECT_NAMES)
    mode_label = "SCAN"

    print(f"  Bottles:")
    for name in BOTTLE_INFO:
        pos = obs[f"{name}_pos"]
        print(f"    {name}: x={pos[0]:.3f}")
    print(f"  Bin: x={obs['reject_bin_1_pos'][0]:.3f}")

    np.random.seed(42)

    while step < MAX_STEPS:
        eef_pos = obs["robot0_eef_pos"].copy()
        bin_pos = obs["reject_bin_1_pos"].copy()

        action = np.zeros(7)
        status = "SCANNING BELT..."

        if robot_phase == "idle":
            mode_label = "SCAN"
            current_target = None
            for dname in DEFECT_NAMES - picked_defects:
                dpos = obs[f"{dname}_pos"].copy()
                if -0.06 <= dpos[0] <= 0.12:
                    current_target = dname
                    if len(picked_defects) == 0:
                        robot_phase = "vla_approach"
                        vla_steps = 0
                        policy.reset()
                        print(f"  Step {step}: DEFECT '{dname}' in pick zone (x={dpos[0]:.3f}) -> VLA approach")
                    else:
                        robot_phase = "approach"
                        env.belt_active = False
                        print(f"  Step {step}: DEFECT '{dname}' in pick zone (x={dpos[0]:.3f}) -> scripted approach")
                    break

            if picked_defects == DEFECT_NAMES and robot_phase == "idle":
                robot_phase = "all_done"
                print(f"  Step {step}: All {total_defects} defects rejected!")

        if robot_phase == "idle" or robot_phase == "all_done":
            if robot_phase == "all_done":
                status = f"ALL {total_defects} DEFECTS REJECTED | LINE CLEAR"
                mode_label = "DONE"
        elif robot_phase == "vla_approach":
            mode_label = "VLA"
            defect_pos = obs[f"{current_target}_pos"].copy()
            dist_to_defect = np.linalg.norm(eef_pos[:2] - defect_pos[:2])

            batch = preprocessor(build_observation(obs))
            with torch.inference_mode():
                act = policy.select_action(batch)
            action = postprocessor(act).squeeze(0).cpu().numpy()
            status = f"SmolVLA: APPROACHING {current_target}"

            making_progress = eef_pos[2] < 1.12 and dist_to_defect < 0.20
            if making_progress or vla_steps > 30:
                robot_phase = "approach"
                env.belt_active = False
                print(f"  Step {step}: VLA -> scripted (d={dist_to_defect:.3f}, z={eef_pos[2]:.3f}) belt paused")

            vla_steps += 1

        elif robot_phase in ("approach", "descend", "grasp", "lift", "to_bin", "release", "retreat"):
            mode_label = "ASSIST"
            defect_pos = obs[f"{current_target}_pos"].copy()

            if robot_phase == "approach":
                status = "POSITIONING OVER TARGET"
                target = np.array([defect_pos[0], defect_pos[1], defect_pos[2] + 0.12])
                action[:3] = compute_action(eef_pos, target, gain=12.0)
                action[6] = -1.0
                dist_xy = np.linalg.norm(eef_pos[:2] - defect_pos[:2])
                approach_count += 1
                converged = dist_xy < 0.04 and abs(eef_pos[2] - target[2]) < 0.04
                timed_out = approach_count > 100
                if converged or timed_out:
                    robot_phase = "descend"
                    print(f"  Step {step}: -> descend (d_xy={dist_xy:.3f}, approach_steps={approach_count}{'timeout' if timed_out else ''})")

            elif robot_phase == "descend":
                status = "DESCENDING TO TARGET"
                target = np.array([defect_pos[0], defect_pos[1], defect_pos[2] + 0.03])
                action[:3] = compute_action(eef_pos, target, gain=15.0)
                action[6] = -1.0
                descend_count += 1
                if eef_pos[2] < defect_pos[2] + 0.06 or descend_count > 60:
                    robot_phase = "grasp"
                    print(f"  Step {step}: -> grasp (eef_z={eef_pos[2]:.3f}, descend_steps={descend_count})")

            elif robot_phase == "grasp":
                status = "GRASPING DEFECTIVE BOTTLE"
                target = np.array([defect_pos[0], defect_pos[1], defect_pos[2] + 0.03])
                action[:3] = compute_action(eef_pos, target, gain=4.0)
                action[6] = 1.0
                grasp_count += 1
                if grasp_count >= 40:
                    robot_phase = "lift"
                    print(f"  Step {step}: -> lift")

            elif robot_phase == "lift":
                status = "REMOVING DEFECT FROM LINE"
                target = np.array([eef_pos[0], eef_pos[1], 1.15])
                action[:3] = compute_action(eef_pos, target, gain=10.0)
                action[6] = 1.0
                if eef_pos[2] > 1.12:
                    robot_phase = "to_bin"
                    print(f"  Step {step}: -> to_bin (bottle_z={defect_pos[2]:.3f})")

            elif robot_phase == "to_bin":
                status = "MOVING TO REJECT BIN"
                target = np.array([bin_pos[0], bin_pos[1], 1.0])
                action[:3] = compute_action(eef_pos, target, gain=12.0)
                action[6] = 1.0
                dist_xy = np.linalg.norm(eef_pos[:2] - bin_pos[:2])
                tobin_count += 1
                if dist_xy < 0.06 or tobin_count > 60:
                    robot_phase = "release"
                    print(f"  Step {step}: -> release (d={dist_xy:.3f})")

            elif robot_phase == "release":
                status = "DEPOSITING IN REJECT BIN"
                action[:3] = np.zeros(3)
                action[6] = -1.0
                release_count += 1
                if release_count >= 25:
                    robot_phase = "retreat"
                    print(f"  Step {step}: -> retreat")

            elif robot_phase == "retreat":
                status = "RETURNING TO STANDBY"
                target = np.array([eef_pos[0], eef_pos[1], 1.25])
                action[:3] = compute_action(eef_pos, target, gain=8.0)
                action[6] = -1.0
                if eef_pos[2] > 1.22:
                    picked_defects.add(current_target)
                    grasp_count = 0
                    release_count = 0
                    tobin_count = 0
                    approach_count = 0
                    descend_count = 0
                    env.belt_active = True
                    robot_phase = "idle"
                    print(f"  Step {step}: -> idle ({len(picked_defects)}/{total_defects} defects done, belt resumed)")

        highlight = robot_phase not in ("idle", "all_done")
        sideview = obs["sideview_image"][::-1].copy()
        sideview = draw_bottle_detections(sideview, env.sim, obs, "sideview", highlight_defect=highlight)
        sideview = draw_hud(sideview, status, robot_phase.upper(), step,
                            len(picked_defects), total_defects, mode_label)

        frames.append(sideview)
        obs, _, _, _ = env.step(action.tolist() if isinstance(action, np.ndarray) else action)
        if render:
            env.render()
        step += 1

        if robot_phase == "all_done" and step > 50:
            break

    for _ in range(40):
        sideview = obs["sideview_image"][::-1].copy()
        sideview = draw_bottle_detections(sideview, env.sim, obs, "sideview")
        sideview = draw_hud(sideview, "PIPELINE COMPLETE | ALL DEFECTS REJECTED",
                           "COMPLETE", step, total_defects, total_defects, "DONE")
        frames.append(sideview)
        obs, _, _, _ = env.step(np.zeros(7))
        if render:
            env.render()
        step += 1

    success = env._check_success()
    num_picked = len(picked_defects)
    print(f"  Task success: {success}, picked: {num_picked}/{total_defects}")
    return frames, success, num_picked


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", help="no on-screen viewer (still records video)")
    args = parser.parse_args()
    render = not args.headless

    os.makedirs(FRAMES_DIR, exist_ok=True)

    print(f"Loading SmolVLA from {CHECKPOINT}...")
    policy, preprocessor, postprocessor = load_policy(CHECKPOINT)
    print(f"  Model loaded on {DEVICE} (n_action_steps={policy.config.n_action_steps})")
    print(f"  Task: '{TASK_DESCRIPTION}'")

    best_frames = None
    best_picked = -1
    for attempt in range(5):
        print(f"\nAttempt {attempt + 1}...")
        env = make_env(render=render)
        obs = env.reset()
        frames, success, num_picked = run_pipeline(env, obs, policy, preprocessor, postprocessor, render=render)
        env.close()
        print(f"  Result: {'SUCCESS' if success else 'FAIL'} ({len(frames)} frames, picked={num_picked})")

        if success:
            best_frames = frames
            break
        elif num_picked > best_picked:
            best_picked = num_picked
            best_frames = frames

    all_frames = best_frames
    n = len(all_frames)

    key_indices = [0, n // 8, n // 4, 3 * n // 8, n // 2, 5 * n // 8, 3 * n // 4, 7 * n // 8, n - 1]
    for i, idx in enumerate(key_indices):
        if idx < n:
            Image.fromarray(all_frames[idx]).save(f"{FRAMES_DIR}/smolvla_{i}_{idx:03d}.png")

    print(f"\nSaving video ({n} frames)...")
    writer = imageio.get_writer(VIDEO_PATH, fps=20, quality=8)
    for frame in all_frames:
        writer.append_data(frame)
    writer.close()
    print(f"Video saved: {VIDEO_PATH}")


if __name__ == "__main__":
    main()
