"""Render a labelled dataset for the bottle detector (overhead RGB-D camera).

    python -m inspection.generate_detection_data --out detection_data/train --n 3000 --seed 0
    python -m inspection.generate_detection_data --out detection_data/val   --n 400  --seed 1

Each sample is one image from the overhead tracking camera (env.TRACK_CAMERA) with a box
around every visible bottle and its class (upright or fallen). Boxes come from the
simulator's segmentation render, so they cover exactly the visible pixels (a bottle half
hidden by the arm gets the box of its visible half; one hidden almost entirely gets none).
Simulator positions are used here only to create labels; at run time the detector works from
the image alone.

Per sample: a fresh mix of bottle designs every RESET_EVERY samples; bottles at random spots
on the belt (some lying down, some already in the reject box, some out of view), random
looks and defects, random arm pose (it occludes bottles), light changes and a little camera
pose jitter; then the camera model (sensor noise + JPEG). Writes images/*.jpg and
annotations.json (boxes, labels and, for evaluation, each bottle's true position).
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from scipy import ndimage
import robosuite.utils.transform_utils as T
from PIL import Image

import pipeline_demo as line
from inspection.station import JPEG_QUALITY, SENSOR_NOISE, add_sensor_noise
from inspection.vision import BELT_Z, LABELS

RESET_EVERY = 25
VIEW_X = (-0.17, 0.60)      # belt stretch the overhead camera sees
FAR_X = 0.86                # parking spot out of view (upstream)
MIN_VISIBLE_PX = 40


def place(env, name, pos, quat_wxyz):
    joint = env.objects_dict[name].joints[-1]
    q = env.sim.data.get_joint_qpos(joint).copy()
    q[:3], q[3:7] = pos, quat_wxyz
    env.sim.data.set_joint_qpos(joint, q)


def yaw_quat(yaw):
    return np.array([np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)])


def lying_quat(rng):
    """Bottle axis horizontal, pointing in a random direction."""
    yaw = rng.uniform(0, 2 * np.pi)
    tip = T.axisangle2quat(np.array([0.0, np.pi / 2, 0.0]))                 # xyzw: z axis -> x
    q = T.quat_multiply(T.axisangle2quat(np.array([0.0, 0.0, yaw])), tip)
    return T.convert_quat(q, to="wxyz")


class Scene:
    """Randomizes bottle placement, looks, arm pose, lights and the overhead camera pose."""

    def __init__(self, env):
        m = env.sim.model
        self.env = env
        self.light_diffuse = m.light_diffuse.copy()
        self.light_pos = m.light_pos.copy()
        self.cam = m.camera_name2id(env.TRACK_CAMERA)
        self.cam_pos = m.cam_pos[self.cam].copy()
        self.cam_quat = m.cam_quat[self.cam].copy()
        robot = env.robots[0]
        self.arm_idx = robot._ref_joint_pos_indexes
        self.arm_home = env.sim.data.qpos[self.arm_idx].copy()
        self.bin_xy = None

    def randomize(self, rng, obs):
        env, m = self.env, self.env.sim.model
        shapes = env.bottle_shapes
        if self.bin_xy is None:
            self.bin_xy = np.array(obs["reject_bin_1_pos"][:2])
            self.bin_z = float(obs["reject_bin_1_pos"][2])
        placed = []                       # (x, y, r) of bottles on the belt
        bin_slots = [np.array([-0.035, 0.0]), np.array([0.035, 0.0])]
        rng.shuffle(bin_slots)
        for name in rng.permutation(env.inspection_bottles):
            s = shapes[name]
            defect = str(rng.choice(s.defects)) if rng.random() < 0.35 else None
            env.set_bottle_defect(name, defect, rng)
            u = rng.random()
            if u < 0.08 and bin_slots:                                   # already in the reject box
                xy = self.bin_xy + bin_slots.pop()
                place(env, name, [xy[0], xy[1], self.bin_z + 0.006], yaw_quat(rng.uniform(0, 2 * np.pi)))
                continue
            if u > 0.92:                                                 # out of view
                place(env, name, [FAR_X + 0.15 * len(placed), 0.0, BELT_Z], yaw_quat(0.0))
                continue
            fallen = rng.random() < 0.12
            r = s.r_body if not fallen else s.height / 2
            for _ in range(50):
                x, y = rng.uniform(*VIEW_X), rng.uniform(-0.09, 0.09) if not fallen else rng.uniform(-0.06, 0.06)
                if all(np.hypot(x - px, y - py) > r + pr + 0.006 for px, py, pr in placed):
                    break
            else:
                place(env, name, [FAR_X + 0.15 * len(placed), 0.3, BELT_Z], yaw_quat(0.0))
                continue
            placed.append((x, y, r))
            if fallen:
                q = lying_quat(rng)
                axis = T.quat2mat(T.convert_quat(q, to="xyzw"))[:, 2]
                base = np.array([x, y, BELT_Z + s.r_body]) - axis * s.height / 2   # body centred on (x, y)
                place(env, name, base, q)
            else:
                place(env, name, [x, y, BELT_Z], yaw_quat(rng.uniform(0, 2 * np.pi)))
        # arm somewhere around its home pose (often over the belt, hiding bottles)
        qpos = env.sim.data.qpos
        qpos[self.arm_idx] = self.arm_home + rng.uniform(-0.6, 0.6, len(self.arm_idx)) * (rng.random() < 0.7)
        # lights and camera pose
        m.light_diffuse[:] = np.clip(self.light_diffuse * rng.uniform(0.6, 1.4, (len(self.light_diffuse), 1)), 0, 1)
        m.light_pos[:] = self.light_pos + rng.uniform(-0.08, 0.08, self.light_pos.shape)
        m.cam_pos[self.cam] = self.cam_pos + rng.uniform(-0.004, 0.004, 3)
        jitter = T.axisangle2quat(rng.normal(0, np.radians(0.5), 3))
        m.cam_quat[self.cam] = T.convert_quat(T.quat_multiply(T.convert_quat(self.cam_quat, to="xyzw"), jitter), to="wxyz")
        env.sim.forward()


def render_segmentation(sim, camera, w, h):
    """(H, W, 2) [object type, object id] per pixel, upright. robosuite's own decoder adds the
    uint8 colour channels without widening them first, which overflows with NumPy 2."""
    ctx = sim._render_context_offscreen
    ctx.render(width=w, height=h, camera_id=sim.model.camera_name2id(camera), segmentation=True)
    rgb = np.empty((h, w, 3), dtype=np.uint8)
    mujoco.mjr_readPixels(rgb=rgb, depth=None, viewport=mujoco.MjrRect(0, 0, w, h), con=ctx.con)
    seg = rgb[..., 0].astype(np.int64) + rgb[..., 1].astype(np.int64) * 256 + rgb[..., 2].astype(np.int64) * 65536
    n = ctx.scn.ngeom
    seg[seg >= n + 1] = 0
    ids = np.full((n + 1, 2), -1, dtype=np.int64)
    for i in range(n):
        g = ctx.scn.geoms[i]
        if g.segid != -1:
            ids[g.segid + 1] = (g.objtype, g.objid)
    return ids[seg][::-1]


def bottle_masks(env, seg):
    """name -> boolean mask of the bottle's visible pixels, from a segmentation render."""
    m = env.sim.model
    roots = {m.body_rootid[env.obj_body_id[n]]: n for n in env.inspection_bottles}
    geom_ids = seg[..., 1]
    is_geom = seg[..., 0] == int(mujoco.mjtObj.mjOBJ_GEOM)
    owner = np.full(m.ngeom, "", dtype=object)
    for g in range(m.ngeom):
        owner[g] = roots.get(m.body_rootid[m.geom_bodyid[g]], "")
    names = np.where(is_geom & (geom_ids >= 0), owner[np.clip(geom_ids, 0, m.ngeom - 1)], "")
    return {n: clean_mask(names == n) for n in env.inspection_bottles}


def clean_mask(mask, min_component=12):
    """Drop stray pixels: anti-aliasing blends neighbouring ID colours at geom edges into
    unrelated IDs, which would otherwise stretch the box across the image."""
    labels, n = ndimage.label(mask)
    if n == 0:
        return mask
    sizes = ndimage.sum(mask, labels, index=np.arange(1, n + 1))
    keep = np.flatnonzero(sizes >= min_component) + 1
    return np.isin(labels, keep)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", required=True)
    p.add_argument("--n", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    img_dir = os.path.join(args.out, "images")
    os.makedirs(img_dir, exist_ok=True)
    env = line.make_env(record=False)
    h, w = env.TRACK_CAM_SIZE
    records = []
    for i in range(args.n):
        if i % RESET_EVERY == 0:
            obs = env.reset()
            env.belt_active = False
            scene = Scene(env)
        scene.randomize(rng, obs)
        rgb = env.sim.render(width=w, height=h, camera_name=env.TRACK_CAMERA)[::-1]
        seg = render_segmentation(env.sim, env.TRACK_CAMERA, w, h)
        boxes, labels, truth = [], [], []
        for name, mask in bottle_masks(env, seg).items():
            if mask.sum() < MIN_VISIBLE_PX:
                continue
            ys, xs = np.nonzero(mask)
            rot = env.sim.data.body_xmat[env.obj_body_id[name]].reshape(3, 3)
            upright = rot[2, 2] > 0.7
            pos = env.sim.data.body_xpos[env.obj_body_id[name]]
            boxes.append([int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1])
            labels.append(LABELS.index("upright" if upright else "fallen"))
            s = env.bottle_shapes[name]
            truth.append(dict(name=name, design=s.name, base=[float(v) for v in pos],
                              top_z=float(pos[2] + s.height) if upright else None,
                              in_bin=bool(np.linalg.norm(pos[:2] - scene.bin_xy) < 0.08)))
        img = add_sensor_noise(rgb, rng, SENSOR_NOISE)
        fn = f"{i:06d}.jpg"
        Image.fromarray(img).save(os.path.join(img_dir, fn), quality=JPEG_QUALITY)
        records.append(dict(file=fn, boxes=boxes, labels=labels, truth=truth))
        if (i + 1) % 250 == 0:
            print(f"  {i + 1}/{args.n}", flush=True)
    with open(os.path.join(args.out, "annotations.json"), "w") as f:
        json.dump(dict(image_size=[h, w], labels=list(LABELS), samples=records), f)
    env.close()
    print("wrote", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
