"""Render a labelled bottle-inspection dataset from the simulator.

    python -m inspection.generate_dataset --out inspection_data/train --n 7000 --seed 0
    python -m inspection.generate_dataset --out inspection_data/val   --n 1000 --seed 1
    python -m inspection.generate_dataset --out inspection_data/test  --n 2500 --seed 2
    python -m inspection.generate_dataset --out inspection_data/stress --n 1500 --seed 3 --stress

Each sample is the pair of photos the inspection station would take (same cameras, same
render call), of one bottle at the station. Half the samples are good bottles; the
other half are spread evenly over the defect types (crack and chip only occur on glass
designs, dented only on PET). Every RESET_EVERY samples the scene is rebuilt, which draws a
new mix of bottle designs. Per sample the script randomizes: the bottle's colours, look and
defect parameters (including harmless bubbles, droplets and label shifts), its yaw and
position (trigger jitter and across-belt offset), neighbouring bottles, light intensities
and positions, and a few millimetres / a degree of camera pose. Images go through the station's camera model (sensor noise + JPEG). Writes
images/*.jpg and labels.csv. --stress widens the lighting and camera randomization beyond
the training range and doubles the sensor noise, to measure robustness.

--station pick renders from the close-up cameras at the pick area (env.PICK_CAMERAS, 512 px
by default) instead, for training a policy that judges the bottle it is about to pick. The
bottle is placed anywhere in the stretch where it is judged on its way in (PICK_WINDOW), some
samples have no bottle in that stretch (label "none"), and the hard defects (crack, chip,
particle) get more samples. Example:

    python -m inspection.generate_dataset --station pick --out pick_data/train --n 20000 --seed 10
"""
import argparse
import csv
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import robosuite.utils.transform_utils as T
from PIL import Image

import pipeline_demo as line
from inspection.model import CLASSES, IMAGE_SIZE
from inspection.station import JPEG_QUALITY, SENSOR_NOISE, add_sensor_noise, render_views

BELT_Z = 0.9
FAR_X = 0.86        # parking spot for bottles that should be out of view
RESET_EVERY = 40    # samples per scene build (each build draws new bottle designs)
PICK_WINDOW = (0.02, 0.10)   # x stretch in which a bottle arriving at the pick area is judged
DEFECT_WEIGHTS = {"crack": 1.4, "chip": 1.4, "particle": 1.4, "missing_cap": 0.7}   # pick station only


def set_pose(env, name, x, y, yaw):
    joint = env.objects_dict[name].joints[-1]
    q = env.sim.data.get_joint_qpos(joint).copy()
    q[:3] = [x, y, BELT_Z]
    q[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    env.sim.data.set_joint_qpos(joint, q)


class DomainRandomizer:
    """Small random changes to lights and inspection-camera poses, undone by restore()."""

    def __init__(self, env, strength=1.0, cameras=None):
        m = env.sim.model
        self.env = env
        self.k = strength
        self.light_diffuse = m.light_diffuse.copy()
        self.light_pos = m.light_pos.copy()
        self.cams = [m.camera_name2id(c) for c in (cameras or env.INSPECTION_CAMERAS)]
        self.cam_pos = m.cam_pos[self.cams].copy()
        self.cam_quat = m.cam_quat[self.cams].copy()

    def apply(self, rng):
        m = self.env.sim.model
        lo, hi = 1 - 0.3 * self.k, 1 + 0.3 * self.k
        m.light_diffuse[:] = np.clip(self.light_diffuse * rng.uniform(lo, hi, (len(self.light_diffuse), 1)), 0, 1)
        m.light_pos[:] = self.light_pos + rng.uniform(-0.04, 0.04, self.light_pos.shape) * self.k
        for k, c in enumerate(self.cams):
            m.cam_pos[c] = self.cam_pos[k] + rng.uniform(-0.003, 0.003, 3) * self.k
            jitter = T.axisangle2quat(rng.normal(0, np.radians(0.7) * self.k, 3))          # xyzw
            q = T.quat_multiply(T.convert_quat(self.cam_quat[k], to="xyzw"), jitter)
            m.cam_quat[c] = T.convert_quat(q, to="wxyz")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--good-fraction", type=float, default=0.5)
    p.add_argument("--stress", action="store_true", help="harsher lighting/camera randomization and 2x sensor noise")
    p.add_argument("--station", choices=("inspection", "pick"), default="inspection",
                   help="which camera pair: the inspection station or the close-up cameras at the pick area")
    p.add_argument("--size", type=int, default=None, help="image size (default 320 inspection, 512 pick)")
    p.add_argument("--none-fraction", type=float, default=0.06, help="pick station: samples with no bottle to judge")
    args = p.parse_args()
    pick = args.station == "pick"
    size = args.size or (512 if pick else IMAGE_SIZE)

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)

    env = line.make_env(record=False)
    noise = 2 * SENSOR_NOISE if args.stress else SENSOR_NOISE
    defects = CLASSES[1:]
    weights = np.array([DEFECT_WEIGHTS.get(d, 1.0) if pick else 1.0 for d in defects])
    cameras = env.PICK_CAMERAS if pick else env.INSPECTION_CAMERAS

    with open(os.path.join(args.out, "labels.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["idx", "label", "front", "back", "design", "x"])
        for i in range(args.n):
            if i % RESET_EVERY == 0:
                env.reset()                       # hard reset: new bottle designs
                env.belt_active = False
                rand = DomainRandomizer(env, strength=2.0 if args.stress else 1.0, cameras=cameras)
                bottles = env.inspection_bottles
                shapes = env.bottle_shapes
            if pick and rng.random() < args.none_fraction:
                label = "none"           # nothing in the judging stretch: bottles just before and after it
                a, b, *rest = rng.permutation(bottles)
                set_pose(env, a, PICK_WINDOW[1] + rng.uniform(0.03, 0.10), rng.uniform(-0.03, 0.03), rng.uniform(0, 2 * np.pi))
                set_pose(env, b, PICK_WINDOW[0] - rng.uniform(0.02, 0.12), rng.uniform(-0.03, 0.03), rng.uniform(0, 2 * np.pi))
                for name in (a, b):
                    env.set_bottle_defect(name, str(rng.choice(shapes[name].defects)) if rng.random() < 0.3 else None, rng)
                for k, name in enumerate(rest):
                    set_pose(env, name, FAR_X + 0.1 * k, 0.0, 0.0)
                rand.apply(rng)
                env.sim.forward()
                views = [add_sensor_noise(v, rng, noise) for v in render_views(env, size, cameras)]
                names = (f"{i:06d}_front.jpg", f"{i:06d}_back.jpg")
                for img, fn in zip(views, names):
                    Image.fromarray(img).save(os.path.join(img_dir, fn), quality=JPEG_QUALITY)
                writer.writerow([i, label, *names, "", ""])
                continue
            while True:
                label = "good" if rng.random() < args.good_fraction else str(rng.choice(defects, p=weights / weights.sum()))
                able = [b for b in bottles if label == "good" or label in shapes[b].defects]
                if able:
                    break
            target = str(rng.choice(able))
            up, down, *rest = rng.permutation([b for b in bottles if b != target])
            env.set_bottle_defect(target, None if label == "good" else label, rng)
            x0 = rng.uniform(*PICK_WINDOW) if pick else env.INSPECTION_X + rng.uniform(-0.003, 0.0)
            set_pose(env, target, x0, rng.uniform(-0.03, 0.03), rng.uniform(0, 2 * np.pi))
            for neighbour, sign in ((up, 1), (down, -1)):     # bottles before/after it on the belt
                env.set_bottle_defect(neighbour, str(rng.choice(shapes[neighbour].defects)) if rng.random() < 0.3 else None, rng)
                set_pose(env, neighbour, x0 + sign * rng.uniform(0.10, 0.16), rng.uniform(-0.03, 0.03), rng.uniform(0, 2 * np.pi))
            for k, name in enumerate(rest):
                set_pose(env, name, FAR_X + 0.1 * k, 0.0, 0.0)
            rand.apply(rng)
            env.sim.forward()
            # Same camera model as InspectionStation.capture: sensor noise, then JPEG.
            views = [add_sensor_noise(v, rng, noise) for v in render_views(env, size, cameras)]
            names = (f"{i:06d}_front.jpg", f"{i:06d}_back.jpg")
            for img, fn in zip(views, names):
                Image.fromarray(img).save(os.path.join(img_dir, fn), quality=JPEG_QUALITY)
            writer.writerow([i, label, *names, shapes[target].name, f"{x0:.4f}"])
            if (i + 1) % 500 == 0:
                print(f"  {i + 1}/{args.n}", flush=True)
    env.close()
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
