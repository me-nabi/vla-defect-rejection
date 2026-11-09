"""Evaluate the bottle detector and 3D localisation on fresh simulated scenes.

    python -m inspection.evaluate_detector --n 300 --seed 7

Builds random scenes the same way as generate_detection_data (unseen seed), runs the
overhead camera (with its noise) -> detector -> depth localisation exactly as on the line,
and compares with the true bottle poses:
  * recall: bottles on the belt that are mostly visible (>= 60 % of their unoccluded pixel
    area) and are found within 3 cm
  * precision: detections on the belt that correspond to a real bottle
  * position error of the grasp point (axis xy) and height error, for upright bottles
  * upright/fallen accuracy
Writes inspection_models/detector_eval.json.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import pipeline_demo as line
from inspection.generate_detection_data import RESET_EVERY, Scene, bottle_masks, render_segmentation
from inspection.vision import BELT_Z, BottleDetector, OverheadCamera, localize

MATCH_DIST = 0.03


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=os.path.join(REPO, "inspection_models", "detector_eval.json"))
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)
    detector = BottleDetector(line.DETECTOR_PATH)
    env = line.make_env(record=False)
    h, w = env.TRACK_CAM_SIZE
    found = total = det_on_belt = det_true = 0
    xy_err, h_err, cls_ok, cls_n = [], [], 0, 0
    per_design = {}
    for i in range(args.n):
        if i % RESET_EVERY == 0:
            obs = env.reset()
            env.belt_active = False
            scene = Scene(env)
            camera = OverheadCamera(env, rng)
        scene.randomize(rng, obs)
        seg = render_segmentation(env.sim, env.TRACK_CAMERA, w, h)
        masks = bottle_masks(env, seg)
        rgb, depth = camera.capture()
        dets = [localize(camera, depth, b, s, l) for b, s, l in detector(rgb)]
        dets = [d for d in dets if d is not None]

        truth = []
        for name in env.inspection_bottles:
            pos = env.sim.data.body_xpos[env.obj_body_id[name]].copy()
            rot = env.sim.data.body_xmat[env.obj_body_id[name]].reshape(3, 3)
            s = env.bottle_shapes[name]
            upright = rot[2, 2] > 0.7
            on_belt = abs(pos[2] - BELT_Z) < 0.05 and np.linalg.norm(pos[:2] - scene.bin_xy) > 0.12
            if not on_belt and upright:
                continue
            centre = pos[:2] if upright else (pos + rot[:, 2] * s.height / 2)[:2]
            # visible fraction: compare with the area of the bottle's footprint seen from above
            full_area = np.pi * (s.r_body * 1.15) ** 2 if upright else 2 * s.r_body * s.height
            px_per_m = (h / 2) / np.tan(np.radians(env.TRACK_CAM_FOVY) / 2) / (1.55 - BELT_Z - (s.height if upright else 2 * s.r_body))
            vis = masks[name].sum() / (full_area * px_per_m ** 2)
            truth.append(dict(name=name, design=s.name, xy=centre, top=pos[2] + s.height if upright else None,
                              upright=upright, visible=vis, on_belt=on_belt))
        taken = set()
        for t in truth:
            if not t["on_belt"] or t["visible"] < 0.6:
                continue
            total += 1
            pd = per_design.setdefault(t["design"], dict(bottles=0, found=0, xy_err=[], h_err=[]))
            pd["bottles"] += 1
            dists = [np.linalg.norm(d.xy - t["xy"]) if k not in taken else 9 for k, d in enumerate(dets)]
            if dists and min(dists) < MATCH_DIST:
                k = int(np.argmin(dists))
                taken.add(k)
                found += 1
                pd["found"] += 1
                d = dets[k]
                cls_n += 1
                cls_ok += d.upright == t["upright"]
                if t["upright"] and d.upright:
                    xy_err.append(dists[k])
                    h_err.append(abs(d.top_z - t["top"]))
                    pd["xy_err"].append(dists[k])
                    pd["h_err"].append(abs(d.top_z - t["top"]))
        for d in dets:                       # precision: detections on the belt (outside the reject box)
            if np.linalg.norm(d.xy - scene.bin_xy) < 0.12:
                continue
            det_on_belt += 1
            det_true += any(np.linalg.norm(d.xy - t["xy"]) < MATCH_DIST for t in truth if t["on_belt"])
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{args.n}", flush=True)
    env.close()

    mm = lambda v: round(1000 * float(v), 1)
    report = dict(
        scenes=args.n, bottles=total, recall=found / max(total, 1), precision=det_true / max(det_on_belt, 1),
        upright_fallen_accuracy=cls_ok / max(cls_n, 1),
        xy_error_mm=dict(mean=mm(np.mean(xy_err)), p95=mm(np.percentile(xy_err, 95)), max=mm(np.max(xy_err))),
        height_error_mm=dict(mean=mm(np.mean(h_err)), p95=mm(np.percentile(h_err, 95)), max=mm(np.max(h_err))),
        per_design={k: dict(bottles=v["bottles"], recall=v["found"] / max(v["bottles"], 1),
                            xy_error_mm_mean=mm(np.mean(v["xy_err"])) if v["xy_err"] else None,
                            height_error_mm_mean=mm(np.mean(v["h_err"])) if v["h_err"] else None)
                    for k, v in sorted(per_design.items())},
    )
    print(json.dumps(report, indent=2))
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)
    print("Report:", args.out)


if __name__ == "__main__":
    main()
