"""Record demonstrations of the full inspection line for training SmolVLA.

    python -m data_collection.record_demos --episodes 600                    # one process
    python -m data_collection.record_demos --episodes 600 --workers 3        # parallel

Each episode is one run of the line: five random bottles (designs, colours, 0-2 defects)
ride the belt; when a defective bottle reaches the pick zone the teacher stops the belt,
picks the bottle, drops it in the reject box and restarts the belt. The teacher is the
scripted picker driven by the simulator's ground truth (which bottles are defective and
where they are), so its decisions and grasp targets are always right. The trained detector
and classifier are not involved.

Recorded per step (20 Hz), so one recording can be turned into a dataset for any of the
training stages (pick only, belt control, defect finding) without recording again:
  * cameras, as high-quality H.264 video, upright:
      agentview.mp4  front view  512x512   (SmolVLA input)
      wrist.mp4      wrist view  512x512   (SmolVLA input)
      overhead.mp4   overhead tracking camera 384x512 (detection labels)
  * data.npz: the teacher's action (the label) and the action actually executed (with a
    little noise, see below), robot state, the belt command, the teacher's phase and target,
    the true pose of every bottle, and the full simulator state (qpos/qvel) so any other
    camera, resolution or label can be rendered again later
  * meta.json: seed, bottle designs and defects, picks, outcome, code version

Quality rules:
  * only clean successes are kept: every defective bottle in the reject box, no good bottle
    in it, nothing on the floor and no failed grasp. Failed seeds are logged and skipped.
  * small noise on the executed arm motion (the label stays the teacher's clean action), so
    the data shows how to correct small errors (DART-style) instead of only perfect paths
  * small noise on the grasp target, so grasps vary a little
  * the scene (designs, colours, defects, start poses) is reproducible from the seed
    (np.random.seed(seed) before env.reset()); replaying the actions drifts by ~1-2 mm, so
    re-render from the saved qpos/qvel, which is exact

Output: recordings/<name>/episodes/seed_XXXXXX/ plus recordings/<name>/info.json.
Re-running resumes: seeds already recorded or already failed are skipped.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

FORMAT_VERSION = 1
FPS = 20
SEED_BASE = 100_000          # far away from the seeds used by the evaluation scripts
INSTRUCTION = "inspect the bottles on the conveyor and put every defective bottle in the reject box"
CAMERAS = {                  # file name -> (robosuite camera, height, width)
    "agentview": ("agentview", 512, 512),
    "wrist": ("robot0_eye_in_hand", 512, 512),
    "overhead": ("track_cam", 384, 512),
}
VIDEO_CRF = 18               # visually lossless H.264
DEFECT_COUNT_PROBS = (0.10, 0.45, 0.45)   # 0, 1 or 2 defective bottles (the line's default has 15 % zero)
ACTION_NOISE = 0.05          # std of the noise added to the executed xyz action (units of the [-1, 1] range)
TARGET_NOISE_XY = 0.003      # m, std
TARGET_NOISE_Z = 0.004       # m, std
PAUSE_STEPS = 4              # steps between stopping the belt and starting the pick (as on the line)
SETTLE_STEPS = 30            # steps recorded after the line is clear
MAX_STEPS = 1500
PHASES = ["run", "pause", "approach", "descend", "grasp", "lift", "transfer", "lower", "settle",
          "release", "retreat", "done", "end"]


def git_version():
    try:
        rev = subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
        dirty = bool(subprocess.check_output(["git", "-C", REPO, "status", "--porcelain", "--untracked-files=no"],
                                             text=True).strip())
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def make_env():
    import pipeline_demo as line
    from libero.libero.envs import TASK_MAPPING
    names = [c[0] for c in CAMERAS.values()]
    return TASK_MAPPING["conveyor_belt_manipulation"](
        bddl_file_name=line.BDDL_FILE,
        robots=["Panda"],
        controller_configs=line.CONTROLLER_CONFIG,
        has_renderer=False,
        has_offscreen_renderer=True,
        render_camera="agentview",
        ignore_done=True,
        use_camera_obs=True,
        camera_names=names,
        camera_heights=[c[1] for c in CAMERAS.values()],
        camera_widths=[c[2] for c in CAMERAS.values()],
        reward_shaping=True,
        control_freq=FPS,
        belt_active=True,
        belt_speed=line.BELT_SPEED,
        belt_extension=line.BELT_EXTENSION,
        defect_count_probs=DEFECT_COUNT_PROBS,
    )


class VideoWriters:
    def __init__(self, folder):
        import imageio
        self.writers = {
            name: imageio.get_writer(os.path.join(folder, f"{name}.mp4"), fps=FPS, codec="libx264", quality=None,
                                     pixelformat="yuv420p", macro_block_size=16,
                                     ffmpeg_params=["-crf", str(VIDEO_CRF), "-preset", "fast"])
            for name in CAMERAS}

    def add(self, obs):
        for name, (cam, _, _) in CAMERAS.items():
            self.writers[name].append_data(np.ascontiguousarray(obs[f"{cam}_image"][::-1]))

    def close(self):
        for w in self.writers.values():
            w.close()


def record_episode(env, seed, folder):
    """Run one teacher episode, writing into `folder`. Returns (kept, meta)."""
    from scripted_pick import PICK_ZONE_X, ScriptedPicker

    np.random.seed(seed)
    obs = env.reset()
    rng = np.random.default_rng(seed + 7_919)          # teacher noise, independent of the scene
    names = env.inspection_bottles
    body = [env.obj_body_id[n] for n in names]
    shapes = env.bottle_shapes
    defects = dict(env.bottle_defects)
    bin_xy, bin_z = np.array(obs["reject_bin_1_pos"][:2]), float(obs["reject_bin_1_pos"][2])
    env.belt_active = True

    log = {k: [] for k in ("action", "action_executed", "belt_cmd", "phase", "target", "target_xyz",
                           "joint_pos", "joint_vel", "eef_pos", "eef_quat", "gripper_qpos",
                           "bottle_pos", "bottle_quat", "qpos", "qvel")}
    videos = VideoWriters(folder)
    picker, pick_bottle, pause, handled, picks = None, None, 0, set(), []
    step, settle = 0, None
    try:
        while step < MAX_STEPS:
            pos = env.sim.data.body_xpos[body].copy()
            quat = env.sim.data.body_xquat[body].copy()
            upright = env.sim.data.body_xmat[body].reshape(-1, 3, 3)[:, 2, 2] > 0.7

            if settle is None and picker is None and pause == 0:
                # next defective bottle in the pick zone, most downstream (lowest x) first
                ready = [i for i in np.argsort(pos[:, 0])
                         if defects[names[i]] and names[i] not in handled and upright[i]
                         and PICK_ZONE_X[0] <= pos[i, 0] <= PICK_ZONE_X[1] and abs(pos[i, 2] - 0.90) < 0.03]
                if ready:
                    pick_bottle, pause = int(ready[0]), PAUSE_STEPS
                    env.belt_active = False

            target_xyz = np.full(3, np.nan)
            if pause > 0:
                phase = "pause"
                pause -= 1
                if pause == 0:
                    n = names[pick_bottle]
                    xy = pos[pick_bottle, :2] + rng.normal(0, TARGET_NOISE_XY, 2)
                    top = pos[pick_bottle, 2] + shapes[n].height + rng.normal(0, TARGET_NOISE_Z)
                    picker = ScriptedPicker(dict(label=n, xy=xy, top_z=top), bin_xy, bin_z,
                                            slot=len(picks), log=lambda *_: None)
                    picks.append(dict(bottle=n, defect=defects[n], start_step=step + 1))
            action = np.zeros(7)
            if picker is not None:
                phase = picker.phase
                action = picker.step(obs)
                target_xyz = np.array([*picker.xy, picker.top_z])
                if picker.done:
                    picks[-1].update(end_step=step, attempts=picker.attempt, succeeded=bool(picker.succeeded))
                    handled.add(names[pick_bottle])
                    picker, pick_bottle = None, None
                    env.belt_active = True
            elif pause == 0:
                phase = "run" if settle is None else "end"

            executed = action.copy()
            if phase not in ("run", "pause", "end", "done"):
                executed[:3] = np.clip(executed[:3] + rng.normal(0, ACTION_NOISE, 3), -1, 1)

            videos.add(obs)
            for k, v in (("action", action), ("action_executed", executed), ("belt_cmd", env.belt_active),
                         ("phase", PHASES.index(phase)), ("target", -1 if pick_bottle is None else pick_bottle),
                         ("target_xyz", target_xyz), ("joint_pos", obs["robot0_joint_pos"]),
                         ("joint_vel", obs["robot0_joint_vel"]), ("eef_pos", obs["robot0_eef_pos"]),
                         ("eef_quat", obs["robot0_eef_quat"]), ("gripper_qpos", obs["robot0_gripper_qpos"]),
                         ("bottle_pos", pos), ("bottle_quat", quat),
                         ("qpos", env.sim.data.qpos), ("qvel", env.sim.data.qvel)):
                log[k].append(np.array(v, copy=True))

            obs, _, _, _ = env.step(executed)
            step += 1
            if settle is not None:
                settle -= 1
                if settle <= 0:
                    break
            elif picker is None and pause == 0 and all(
                    obs[f"{n}_pos"][0] < PICK_ZONE_X[0] or obs[f"{n}_pos"][2] < 0.85 for n in names):
                settle = SETTLE_STEPS
    finally:
        videos.close()

    final = env.sim.data.body_xpos[body]
    in_bin = {n: bool(env._eval_predicate(["in", n, "reject_bin_1_contain_region"])) for n in names}
    on_floor = [n for n, p in zip(names, final) if p[2] < 0.5]
    problems = []
    if not env._check_success():
        problems.append("wrong bottles in the reject box")
    if on_floor:
        problems.append("bottle on the floor")
    if any(p.get("attempts", 1) > 1 or not p.get("succeeded", False) for p in picks):
        problems.append("failed grasp")
    if step >= MAX_STEPS:
        problems.append("timeout")
    meta = dict(
        seed=seed, steps=step, instruction=INSTRUCTION, kept=not problems, problems=problems,
        bottles=[dict(name=n, design=shapes[n].name, height=float(shapes[n].height), defect=defects[n],
                      in_reject_box=in_bin[n]) for n in names],
        picks=picks,
    )
    if not problems:
        arrays = {k: np.stack(v) for k, v in log.items()}
        for k in arrays:
            if arrays[k].dtype == np.float64 and k not in ("qpos", "qvel"):
                arrays[k] = arrays[k].astype(np.float32)
        arrays["belt_cmd"] = arrays["belt_cmd"].astype(np.int8)
        arrays["phase"] = arrays["phase"].astype(np.int8)
        arrays["target"] = arrays["target"].astype(np.int8)
        np.savez_compressed(os.path.join(folder, "data.npz"), **arrays)
        with open(os.path.join(folder, "meta.json"), "w") as f:
            json.dump(meta, f, indent=1)
    return not problems, meta


def write_info(root):
    import pipeline_demo as line
    info = dict(
        format_version=FORMAT_VERSION, fps=FPS, instruction=INSTRUCTION, code_version=git_version(),
        cameras={k: dict(camera=c, height=h, width=w, file=f"{k}.mp4") for k, (c, h, w) in CAMERAS.items()},
        video=dict(codec="libx264", crf=VIDEO_CRF, pixel_format="yuv420p", orientation="upright"),
        controller=line.CONTROLLER_CONFIG, belt_speed=line.BELT_SPEED, belt_extension=line.BELT_EXTENSION,
        defect_count_probs=DEFECT_COUNT_PROBS, action_noise=ACTION_NOISE,
        target_noise=dict(xy=TARGET_NOISE_XY, z=TARGET_NOISE_Z), phases=PHASES,
        fields={
            "action": "(T,7) teacher action = training label: OSC_POSE deltas xyz, axis-angle xyz, gripper (-1 open, +1 close)",
            "action_executed": "(T,7) action sent to the simulator: the teacher action plus noise on xyz during picks",
            "belt_cmd": "(T,) 1 = main belt running during this step, 0 = stopped",
            "phase": "(T,) index into `phases`: run/pause/end are teacher states, the rest are picker phases",
            "target": "(T,) index of the bottle being picked (into meta.json bottles), -1 if none",
            "target_xyz": "(T,3) grasp target the teacher used: x, y, top of bottle z (NaN if none)",
            "joint_pos/joint_vel": "(T,7) arm joints", "eef_pos": "(T,3)", "eef_quat": "(T,4) xyzw",
            "gripper_qpos": "(T,2) finger joints",
            "bottle_pos": "(T,5,3) bottle base position in world, order = meta.json bottles",
            "bottle_quat": "(T,5,4) bottle orientation, wxyz",
            "qpos/qvel": "(T,nq)/(T,nv) full MuJoCo state for exact re-rendering (float64)",
        },
        reproduce="np.random.seed(seed); env.reset() with data_collection.record_demos.make_env(), then set qpos/qvel per step",
    )
    with open(os.path.join(root, "info.json"), "w") as f:
        json.dump(info, f, indent=1)


def worker(root, seeds, target, worker_id):
    ep_dir = os.path.join(root, "episodes")
    failed_path = os.path.join(root, f"failed_w{worker_id}.txt")
    failed = set()
    for p in os.listdir(root):
        if p.startswith("failed_w"):
            with open(os.path.join(root, p)) as f:
                failed |= {int(line.split()[0]) for line in f if line.strip()}
    env, kept_now, t0 = None, 0, time.time()
    resets = 0
    for seed in seeds:
        done = len([d for d in os.listdir(ep_dir) if d.startswith("seed_")])
        if done >= target:
            break
        final = os.path.join(ep_dir, f"seed_{seed:06d}")
        if os.path.exists(final) or seed in failed:
            continue
        if env is None or resets >= 40:                 # fresh simulator now and then (memory)
            if env is not None:
                env.close()
            env, resets = make_env(), 0
        tmp = os.path.join(root, "tmp", f"seed_{seed:06d}")
        shutil.rmtree(tmp, ignore_errors=True)
        os.makedirs(tmp)
        t = time.time()
        kept, meta = record_episode(env, seed, tmp)
        resets += 1
        if kept:
            os.rename(tmp, final)
            kept_now += 1
        else:
            shutil.rmtree(tmp, ignore_errors=True)
            with open(failed_path, "a") as f:
                f.write(f"{seed} {'; '.join(meta['problems'])}\n")
        print(f"[w{worker_id}] seed {seed}: {'KEPT' if kept else 'skipped (' + ', '.join(meta['problems']) + ')'}"
              f"  {meta['steps']} steps, {len(meta['picks'])} picks, {time.time() - t:.0f}s"
              f"  | total kept {done + kept}/{target}", flush=True)
    if env is not None:
        env.close()
    print(f"[w{worker_id}] finished: {kept_now} kept in {(time.time() - t0) / 60:.0f} min", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=600, help="kept episodes wanted in total")
    p.add_argument("--name", default="line_v1")
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--seed-base", type=int, default=SEED_BASE)
    args = p.parse_args()

    root = os.path.join(REPO, "recordings", args.name)
    os.makedirs(os.path.join(root, "episodes"), exist_ok=True)
    shutil.rmtree(os.path.join(root, "tmp"), ignore_errors=True)
    write_info(root)
    seeds = range(args.seed_base, args.seed_base + 20 * args.episodes)
    if args.workers == 1:
        worker(root, seeds, args.episodes, 0)
        return
    import multiprocessing as mp
    ctx = mp.get_context("spawn")
    procs = [ctx.Process(target=worker, args=(root, seeds[k::args.workers], args.episodes, k))
             for k in range(args.workers)]
    for pr in procs:
        pr.start()
    for pr in procs:
        pr.join()
    n = len([d for d in os.listdir(os.path.join(root, "episodes")) if d.startswith("seed_")])
    print(f"done: {n} episodes in {root}")


if __name__ == "__main__":
    main()
