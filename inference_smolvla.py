"""
SmolVLA on the conveyor inspection line
=======================================
Five bottles of random designs and colours ride the belt; 0-2 of them have a random defect.
The overhead camera's detector and tracker find the bottles, and the inspection station's
classifier decides which ones to reject (see pipeline_demo.py). For the first flagged
bottle, the SmolVLA policy steers the arm's initial approach before the scripted
pick-and-place takes over; later bottles are handled by the script alone.

    python inference_smolvla.py --headless           # classifier decisions, video only
    python inference_smolvla.py --oracle --seed 3    # ground-truth decisions
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import imageio
import numpy as np
import torch
from PIL import Image

import pipeline_demo as line
from inspection.overlay import LiveWindow, compose_frame, live_frame
from inspection.station import downstream_first
from libero.libero.envs import TASK_MAPPING
from scripted_pick import PICK_ZONE_X, ScriptedPicker

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

CHECKPOINT = os.path.join(REPO, "training_output/smolvla_conveyor/checkpoints/010000/pretrained_model")
VIDEO_PATH = os.path.join(REPO, "smolvla_inference.mp4")
FRAMES_DIR = os.path.join(REPO, "rollout_frames")
MAX_STEPS = line.MAX_STEPS
IMG_H, IMG_W = line.IMG_H, line.IMG_W
CAM_SIZE = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Must match the task string the policy was trained on (lerobot_dataset/meta/tasks.parquet).
TASK_DESCRIPTION = "pick the defective bottle from the conveyor belt and place it in the reject bin"

# Training used chunk_size = n_action_steps = 50, i.e. 50 open-loop actions per
# prediction. Re-plan more often at inference so the policy reacts to the moving belt.
N_ACTION_STEPS = 8

CONTROLLER_CONFIG = line.CONTROLLER_CONFIG


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


def make_env(render=True):
    return TASK_MAPPING["conveyor_belt_manipulation"](
        bddl_file_name=line.BDDL_FILE,
        robots=["Panda"],
        controller_configs=CONTROLLER_CONFIG,
        has_renderer=False,      # live display goes through LiveWindow (annotated), not robosuite's viewer
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        camera_heights=[IMG_H, IMG_H, CAM_SIZE, CAM_SIZE],
        camera_widths=[IMG_W, IMG_W, CAM_SIZE, CAM_SIZE],
        camera_names=[*line.VIDEO_CAMERAS, "agentview", "robot0_eye_in_hand"],
        reward_shaping=True,
        control_freq=20,
        belt_active=True,
        belt_speed=line.BELT_SPEED,
        belt_extension=line.BELT_EXTENSION,
    )


def run_pipeline(env, obs, station, policy, preprocessor, postprocessor, render=True, log=print):
    """One episode. Returns (frames, line success, summary vs ground truth)."""
    vision = line.make_vision(env, obs)
    bin_xy, bin_z = np.array(obs["reject_bin_1_pos"][:2]), float(obs["reject_bin_1_pos"][2])   # cell calibration
    frames, step, picker, n_done = [], 0, None, 0
    mode, target, vla_steps, wait = "scan", None, 0, 0        # scan -> vla -> locate -> assist -> scan
    window = LiveWindow() if render else None
    counter = lambda: f"FLAGGED {len(station.flagged())}  IN BIN {len(line.bottles_in_bin(env))}"

    def show(status, mode_label, active=None):
        if window is not None:
            window.show(live_frame(env, vision, station, "SmolVLA INSPECTION LINE", status, step, counter(),
                                   mode_label, active))

    def start_locate():
        nonlocal mode, wait
        env.belt_active = False
        mode, wait = "locate", line.SNAPSHOT_WAIT

    while step < MAX_STEPS:
        vision.step(step)
        for t in station.update(vision.tracks):
            v = t.verdict
            log(f"  Step {step}: inspected bottle {t.label} ({t.height * 100:.0f} cm tall): "
                f"{'DEFECT - ' + v.reason if v.defective else 'OK'} (p_defect={v.p_defect:.2f})")
        eef_pos = obs["robot0_eef_pos"].copy()
        action = np.zeros(7)
        status = "INSPECTING BOTTLES..."

        if mode == "scan":
            flagged = [t for t in vision.tracks if t.verdict is not None and t.verdict.defective and not t.handled]
            for t in downstream_first(flagged):
                if PICK_ZONE_X[0] <= t.xy[0] <= PICK_ZONE_X[1]:
                    target = t.id
                    if n_done == 0:
                        mode, vla_steps = "vla", 0
                        policy.reset()
                        log(f"  Step {step}: rejecting {t.label} - reason: {t.verdict.reason} -> SmolVLA approach")
                    else:
                        start_locate()
                        log(f"  Step {step}: rejecting {t.label} - reason: {t.verdict.reason} -> scripted pick (belt paused)")
                    break

        track = vision.track(target) if target is not None else None
        reason = station.verdicts[target].reason.upper() if target in station.verdicts else ""
        if mode == "vla":
            if track is None:
                mode, target = "scan", None
            else:
                dist = np.linalg.norm(eef_pos[:2] - track.xy)
                batch = preprocessor(build_observation(obs))
                with torch.inference_mode():
                    action = postprocessor(policy.select_action(batch)).squeeze(0).cpu().numpy()
                status = f"REJECTING bottle #{target}: {reason}  -  SmolVLA approach"
                if (eef_pos[2] < 1.12 and dist < 0.20) or vla_steps > 30:
                    log(f"  Step {step}: SmolVLA -> scripted after {vla_steps} steps (d={dist:.3f}, belt paused)")
                    start_locate()
                vla_steps += 1
        elif mode == "locate":
            status = f"REJECTING bottle #{target}: {reason}  -  LOCATING BOTTLE FOR GRASP"
            wait -= 1
            if wait <= 0:
                vision.detect()
                track = vision.track(target)
                if track is None or not track.upright:
                    log(f"  Step {step}: bottle #{target} {'lost' if track is None else 'is lying down'}; not picking")
                    if track is not None:
                        track.handled = True
                    env.belt_active = True
                    mode, target = "scan", None
                else:
                    picker = ScriptedPicker(dict(label=track.label, xy=track.xy, top_z=track.top_z), bin_xy, bin_z,
                                            slot=n_done, log=log)
                    mode = "assist"
        elif mode == "assist":
            action = picker.step(obs)
            if track is not None:
                track.held = picker.phase in line.CARRY_PHASES
                if track.held:
                    track.follow_gripper(obs["robot0_eef_pos"], picker.hold_dz)
            status = f"REJECTING bottle #{target}: {reason}  -  {picker.status}"
            if picker.done:
                if track is not None:
                    track.handled, track.held = True, False
                n_done += 1
                env.belt_active = True
                log(f"  Step {step}: done with #{target} (belt resumed)")
                mode, picker, target = "scan", None, None

        line_clear = mode == "scan" and all(
            obs[f"{n}_pos"][0] < PICK_ZONE_X[0] or obs[f"{n}_pos"][2] < 0.85 for n in env.inspection_bottles)
        mode_label = {"scan": "SCAN", "vla": "VLA", "locate": "ASSIST", "assist": "ASSIST"}[mode]
        active = target if mode != "scan" else None
        frames.append(render_frame(env, obs, vision, station, status, step, mode_label, active))
        show(status, mode_label, active)
        obs, _, _, _ = env.step(action)
        step += 1
        if line_clear:
            break

    for _ in range(line.TAIL_STEPS):   # film the good bottles going round the bend into the carton
        if not line.bottles_still_moving(env, obs):
            break
        vision.step(step)
        frames.append(render_frame(env, obs, vision, station, "GOOD BOTTLES TO PACKING", step, "SCAN"))
        show("GOOD BOTTLES TO PACKING", "SCAN")
        obs, _, _, _ = env.step(np.zeros(7))
        step += 1
    for _ in range(30):
        obs, _, _, _ = env.step(np.zeros(7))
        step += 1
        outcome = line.outcome_text(station.summary(line.bottles_in_bin(env)))
        frames.append(render_frame(env, obs, vision, station, outcome, step, "DONE"))
        show(outcome, "DONE")

    if window is not None:
        window.close()
    summary = station.summary(line.bottles_in_bin(env))
    success = bool(env._check_success())
    log(f"  Result: {'SUCCESS' if success else 'FAIL'} - {line.outcome_text(summary)}")
    return frames, success, summary


def render_frame(env, obs, vision, station, status, step, mode_label, active=None):
    counter = f"FLAGGED {len(station.flagged())}  IN BIN {len(line.bottles_in_bin(env))}"
    return compose_frame(env, obs, vision.tracks, station, "SmolVLA INSPECTION LINE", status, step, counter,
                         mode_label, active, cams=line.VIDEO_CAMERAS)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--headless", action="store_true", help="no live window (still records video)")
    parser.add_argument("--oracle", action="store_true", help="use ground-truth labels instead of the classifier")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    print(f"Loading SmolVLA from {CHECKPOINT}...")
    policy, preprocessor, postprocessor = load_policy(CHECKPOINT)
    print(f"  Model loaded on {DEVICE} (n_action_steps={policy.config.n_action_steps})")

    if args.seed is not None:
        np.random.seed(args.seed)
    env = make_env(render=not args.headless)
    obs = env.reset()
    station = line.make_station(env, args.oracle)
    print("Bottles this run (ground truth, for reference):",
          {n: (env.bottle_shapes[n].name, env.bottle_defects[n] or "good") for n in env.inspection_bottles})
    frames, success, summary = run_pipeline(env, obs, station, policy, preprocessor, postprocessor,
                                            render=not args.headless)
    env.close()
    print("Summary:", summary)

    os.makedirs(FRAMES_DIR, exist_ok=True)
    n = len(frames)
    for i, idx in enumerate(np.linspace(0, n - 1, 9).astype(int)):
        Image.fromarray(frames[idx]).save(os.path.join(FRAMES_DIR, f"smolvla_{i}_{idx:04d}.png"))
    with imageio.get_writer(VIDEO_PATH, fps=20, quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved: {VIDEO_PATH} ({n} frames)")


if __name__ == "__main__":
    main()
