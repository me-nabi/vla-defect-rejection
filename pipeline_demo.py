"""
Conveyor inspection line with a scripted robot
==============================================
Five bottles ride the belt, each a random design (glass 250/500/330 ml or a PET bottle)
in random colours. On each run 0-2 of them have a random defect: a crack or chip (glass),
a dent (PET), a missing or tilted cap, low fill, or a particle in the drink.

Everything the robot does comes from its own cameras and models:
  1. an overhead RGB-D camera and a trained detector find the bottles; a tracker follows them
     with its own IDs (#1, #2, ...), using the belt speed between frames;
  2. when a track reaches the inspection station, two cameras photograph it and the trained
     classifier decides whether it is defective, and why;
  3. when a flagged bottle reaches the pick zone, the belt pauses, the detector locates it
     again and the robot grasps it at a height that depends on its measured size.
Good bottles continue down the line into the packing carton. The simulator's ground truth is
only used to score the run (or to replace the classifier with --oracle).

    python pipeline_demo.py                # decisions from the trained classifier
    python pipeline_demo.py --oracle       # decisions from the simulator's ground truth
    python pipeline_demo.py --seed 3 --show   # live window with the annotated views
"""
import argparse
import os
import sys

REPO = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, REPO)

import imageio
import numpy as np
from PIL import Image

from inspection.overlay import LiveWindow, compose_frame, live_frame
from inspection.station import InspectionStation, downstream_first
from inspection.vision import BottleDetector, LineVision
from libero.libero.envs import TASK_MAPPING
from scripted_pick import PICK_ZONE_X, ScriptedPicker

BDDL_FILE = os.path.join(REPO, "libero/libero/bddl_files/defect_rejection/multi_bottle_inspection.bddl")
MODEL_PATH = os.path.join(REPO, "inspection_models/inspection_net.pt")
DETECTOR_PATH = os.path.join(REPO, "inspection_models/bottle_detector.pt")
CARRY_PHASES = ("lift", "transfer", "lower", "settle")
SNAPSHOT_WAIT = 4       # steps to wait after pausing the belt before locating the bottle for the grasp
BIN_EXCLUDE_HALF = (0.12, 0.10)   # detections inside the reject box's footprint are not bottles on the belt
VIDEO_PATH = os.path.join(REPO, "pipeline_demo.mp4")
FRAMES_DIR = os.path.join(REPO, "rollout_frames")
BELT_EXTENSION = 0.4    # bottles spawn up to x = 0.86; the default belt ends at x = 0.5
BELT_SPEED = 0.05
MAX_STEPS = 1500
TAIL_STEPS = 450        # when recording, keep filming until good bottles reach the crate (at most this long)
IMG_H, IMG_W = 384, 512
VIDEO_CAMERAS = ("sideview", "lineview")   # close side view + whole-line overview, side by side

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


def make_env(record=True):
    """record=False skips the per-step video cameras (much faster, for evaluation)."""
    return TASK_MAPPING["conveyor_belt_manipulation"](
        bddl_file_name=BDDL_FILE,
        robots=["Panda"],
        controller_configs=CONTROLLER_CONFIG,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=record,
        camera_heights=IMG_H,
        camera_widths=IMG_W,
        camera_names=list(VIDEO_CAMERAS) if record else [],
        reward_shaping=True,
        control_freq=20,
        belt_active=True,
        belt_speed=BELT_SPEED,
        belt_extension=BELT_EXTENSION,
    )


def make_station(env, oracle=False):
    if oracle:
        return InspectionStation(env)
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"No inspection model at {MODEL_PATH}. Train one first "
                         "(python -m inspection.generate_dataset, then python -m inspection.train), "
                         "or run with --oracle.")
    from inspection.model import InspectionClassifier
    return InspectionStation(env, InspectionClassifier(MODEL_PATH))


_DETECTOR = None


def make_vision(env, obs):
    """Overhead camera + trained detector + tracker. The reject box's footprint (cell
    calibration, measured once at start-up) is excluded: bottles in it are already rejected."""
    global _DETECTOR
    if _DETECTOR is None:
        if not os.path.exists(DETECTOR_PATH):
            raise SystemExit(f"No bottle detector at {DETECTOR_PATH}. Train one first "
                             "(python -m inspection.generate_detection_data, then python -m inspection.train_detector).")
        _DETECTOR = BottleDetector(DETECTOR_PATH)
    bin_xy = np.array(obs["reject_bin_1_pos"][:2])
    return LineVision(env, _DETECTOR, exclude=[(bin_xy, BIN_EXCLUDE_HALF)],
                      rng=np.random.default_rng(np.random.randint(2**31)))


def bottles_in_bin(env):
    return [n for n in env.inspection_bottles
            if env._eval_predicate(["in", n, "reject_bin_1_contain_region"])]


def outcome_text(summary):
    problems = []
    if summary["missed_defects"]:
        problems.append(f"{len(summary['missed_defects'])} DEFECT(S) MISSED")
    if summary["rejected_good"]:
        problems.append(f"{len(summary['rejected_good'])} GOOD BOTTLE(S) REJECTED")
    lost = summary["defective"] - summary["rejected_defects"] - len(summary["missed_defects"])
    if lost > 0:
        problems.append(f"{lost} FLAGGED DEFECT(S) NOT IN BIN")
    return "LINE OK: EVERY DEFECT REJECTED, NO GOOD BOTTLE LOST" if not problems else " | ".join(problems)


def render_frame(env, obs, vision, station, status, step, active=None):
    title = "INSPECTION LINE  " + ("(ground truth)" if station.oracle else "(vision model)")
    counter = f"FLAGGED {len(station.flagged())}  IN BIN {len(bottles_in_bin(env))}"
    return compose_frame(env, obs, vision.tracks, station, title, status, step, counter, active=active,
                         cams=VIDEO_CAMERAS)


def bottles_in_crate(env, obs):
    lay = env.belt_layout
    if lay is None:
        return []
    crate_y = lay.exit_y_end - lay.crate_half - 0.012
    return [n for n in env.inspection_bottles
            if abs(obs[f"{n}_pos"][0] - lay.exit_x) < lay.crate_half + 0.01
            and abs(obs[f"{n}_pos"][1] - crate_y) < lay.crate_half + 0.01
            and 0.6 < obs[f"{n}_pos"][2] < 0.85]


def bottles_still_moving(env, obs):
    """Bottles not in the reject bin that are still on a conveyor (not yet in the crate or fallen off)."""
    in_bin = set(bottles_in_bin(env))
    return [n for n in env.inspection_bottles if n not in in_bin and obs[f"{n}_pos"][2] > 0.85]


def run_pipeline(env, obs, station, vision=None, record=True, log=print, show=False):
    """Run one episode. Returns (frames, line success, summary vs ground truth).
    show=True opens a live window with the annotated views."""
    vision = vision or make_vision(env, obs)
    bin_xy, bin_z = np.array(obs["reject_bin_1_pos"][:2]), float(obs["reject_bin_1_pos"][2])   # cell calibration
    frames, step, picker, pending, wait, n_picked = [], 0, None, None, 0, 0
    window = LiveWindow() if show else None
    title = "INSPECTION LINE  " + ("(ground truth)" if station.oracle else "(vision model)")

    def capture(status, active=None):
        if record:
            frames.append(render_frame(env, obs, vision, station, status, step, active))
        if window is not None:
            counter = f"FLAGGED {len(station.flagged())}  IN BIN {len(bottles_in_bin(env))}"
            window.show(live_frame(env, vision, station, title, status, step, counter, active=active))

    while step < MAX_STEPS:
        vision.step(step)
        for t in station.update(vision.tracks):
            v = t.verdict
            log(f"  Step {step}: inspected bottle {t.label} ({t.height * 100:.0f} cm tall): "
                f"{'DEFECT - ' + v.reason if v.defective else 'OK'} (p_defect={v.p_defect:.2f})")
        action = np.zeros(7)
        status = "INSPECTING BOTTLES..."
        active = picker.track_id if picker is not None else (pending if pending is not None else None)

        if picker is None and pending is None:
            flagged = [t for t in vision.tracks if t.verdict is not None and t.verdict.defective and not t.handled]
            for t in downstream_first(flagged):
                if PICK_ZONE_X[0] <= t.xy[0] <= PICK_ZONE_X[1]:
                    env.belt_active = False
                    pending, wait = t.id, SNAPSHOT_WAIT
                    log(f"  Step {step}: bottle {t.label} in the pick zone - reason: {t.verdict.reason} (belt paused)")
                    break

        if pending is not None:
            status = "LOCATING BOTTLE FOR GRASP"
            wait -= 1
            if wait <= 0:
                vision.detect()                     # fresh look with the belt stopped
                t = vision.track(pending)
                if t is None or not t.upright:
                    log(f"  Step {step}: bottle #{pending} {'lost' if t is None else 'is lying down'}; "
                        "not picking (belt resumed)")
                    if t is not None:
                        t.handled = True
                    env.belt_active = True
                else:
                    picker = ScriptedPicker(dict(label=t.label, xy=t.xy, top_z=t.top_z), bin_xy, bin_z,
                                            slot=n_picked, log=log)
                    picker.track_id = t.id
                    log(f"  Step {step}: grasping {t.label} at x={t.xy[0]:.3f} y={t.xy[1]:.3f}, "
                        f"measured height {t.height * 100:.1f} cm")
                pending = None

        if picker is not None:
            action = picker.step(obs)
            t = vision.track(picker.track_id)
            if t is not None:          # the robot knows when it holds the bottle: show it in the gripper
                t.held = picker.phase in CARRY_PHASES
                if t.held:
                    t.follow_gripper(obs["robot0_eef_pos"], picker.hold_dz)
            reason = station.verdicts[picker.track_id].reason.upper()
            status = f"REJECTING bottle #{picker.track_id}: {reason}  -  {picker.status}"
            if picker.done:
                if t is not None:
                    t.handled, t.held = True, False
                n_picked += picker.succeeded
                env.belt_active = True
                log(f"  Step {step}: done with #{picker.track_id} (belt resumed)")
                picker = None

        # The episode ends once no bottle is still upstream of the robot (the scoring harness
        # checks this in the simulator; the robot itself only reacts to what it sees).
        line_clear = picker is None and pending is None and all(
            obs[f"{n}_pos"][0] < PICK_ZONE_X[0] or obs[f"{n}_pos"][2] < 0.85 for n in env.inspection_bottles)
        capture(status, active)
        obs, _, _, _ = env.step(action)
        step += 1
        if line_clear:
            break

    for _ in range(TAIL_STEPS):   # let the good bottles go round the bend into the carton
        if not bottles_still_moving(env, obs):
            break
        vision.step(step)
        capture("GOOD BOTTLES TO PACKING")
        obs, _, _, _ = env.step(np.zeros(7))
        step += 1
    for _ in range(30):     # let the bottles settle, then judge the outcome
        obs, _, _, _ = env.step(np.zeros(7))
        step += 1
        if record or window is not None:
            capture(outcome_text(station.summary(bottles_in_bin(env))))

    if window is not None:
        window.close()
    summary = station.summary(bottles_in_bin(env))
    success = bool(env._check_success())
    good = [n for n, d in env.bottle_defects.items() if d is None]
    crate = set(bottles_in_crate(env, obs))
    summary["good_in_crate"] = sorted(set(good) & crate)
    summary["good_not_delivered"] = sorted(set(good) - crate - set(summary["rejected_good"]))
    summary["on_floor"] = sorted(n for n in env.inspection_bottles if obs[f"{n}_pos"][2] < 0.5)
    summary["steps"] = step
    log(f"  Result: {'SUCCESS' if success else 'FAIL'} - {outcome_text(summary)}")
    return frames, success, summary


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--oracle", action="store_true", help="use ground-truth labels instead of the classifier")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--video", default=VIDEO_PATH)
    parser.add_argument("--show", action="store_true", help="live window with the annotated views")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
    env = make_env()
    obs = env.reset()
    station = make_station(env, args.oracle)
    print("Bottles this run (ground truth, for reference):",
          {n: (env.bottle_shapes[n].name, env.bottle_defects[n] or "good") for n in env.inspection_bottles})
    frames, success, summary = run_pipeline(env, obs, station, show=args.show)
    env.close()
    print("Summary:", summary)

    os.makedirs(FRAMES_DIR, exist_ok=True)
    n = len(frames)
    for i, idx in enumerate(np.linspace(0, n - 1, 9).astype(int)):
        Image.fromarray(frames[idx]).save(os.path.join(FRAMES_DIR, f"pipeline_{i}_{idx:04d}.png"))
    with imageio.get_writer(args.video, fps=20, quality=8) as writer:
        for frame in frames:
            writer.append_data(frame)
    print(f"Video saved: {args.video} ({n} frames)")


if __name__ == "__main__":
    main()
