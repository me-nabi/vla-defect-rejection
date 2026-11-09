"""Evaluate the whole inspection line over many seeded episodes.

    python -m inspection.evaluate_line --episodes 20            # trained classifier + scripted robot
    python -m inspection.evaluate_line --episodes 20 --oracle   # ground-truth decisions (robot only)

The robot finds bottles with its own detector and tracker (inspection/vision.py) in both
modes; --oracle only replaces the classifier's verdicts with the ground truth.

Reports line success (every defect in the bin, no good bottle in it), defects caught and
missed, bottles the vision system never inspected, good bottles wrongly rejected, good
bottles delivered to the packing carton, bottles that ended up on the floor, and results per
bottle design. Writes a JSON report.
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

import pipeline_demo as line


def per_design(rows):
    """Outcome counts per bottle design."""
    out = {}
    for r in rows:
        bad_in_bin = set(r["truth"]) - set(r["missed_defects"])
        for name, design in r["designs"].items():
            d = out.setdefault(design, dict(bottles=0, defective=0, caught=0, good_rejected=0, good_delivered=0))
            d["bottles"] += 1
            if name in r["truth"]:
                d["defective"] += 1
                d["caught"] += name in bad_in_bin
            else:
                d["good_rejected"] += name in r["rejected_good"]
                d["good_delivered"] += name in r["good_in_crate"]
    return dict(sorted(out.items()))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--first-seed", type=int, default=1000, help="seeds first_seed .. first_seed+episodes-1")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--out", default=None, help="JSON report path")
    args = p.parse_args()

    rows = []
    classifier = None
    for i in range(args.episodes):
        seed = args.first_seed + i
        np.random.seed(seed)
        env = line.make_env(record=False)
        obs = env.reset()
        truth = dict(env.bottle_defects)
        designs = {n: s.name for n, s in env.bottle_shapes.items()}
        if args.oracle:
            station = line.make_station(env, oracle=True)
        else:
            if classifier is None:
                from inspection.model import InspectionClassifier
                classifier = InspectionClassifier(line.MODEL_PATH)
            station = line.InspectionStation(env, classifier)
        t0 = time.time()
        _, success, summary = line.run_pipeline(env, obs, station, record=False, log=lambda *a: None)
        env.close()
        row = dict(seed=seed, success=success, truth={k: v for k, v in truth.items() if v}, designs=designs,
                   verdicts={i["truth"]: (i["verdict"].defect if i["verdict"].defective else None,
                                          round(i["verdict"].p_defect, 3)) for i in station.inspections},
                   seconds=round(time.time() - t0, 1), **summary)
        rows.append(row)
        print(f"seed {seed}: {'OK  ' if success else 'FAIL'} defects={row['truth'] or '-'} "
              f"missed={summary['missed_defects'] or '-'} false_flags={summary['false_flags'] or '-'} "
              f"in_bin={summary['rejected_defects']}/{summary['defective']} rejected_good={summary['rejected_good'] or '-'} "
              f"crate={len(summary['good_in_crate'])}/{5 - summary['defective']} floor={summary['on_floor'] or '-'} "
              f"not_inspected={summary['not_inspected'] or '-'} ({row['seconds']}s)",
              flush=True)

    n_def = sum(r["defective"] for r in rows)
    n_good = sum(5 - r["defective"] for r in rows)
    report = dict(
        mode="oracle" if args.oracle else "classifier",
        episodes=len(rows),
        line_success=sum(r["success"] for r in rows),
        defective_bottles=n_def,
        defects_flagged=sum(r["correctly_flagged"] for r in rows),
        defects_rejected=sum(r["rejected_defects"] for r in rows),
        good_bottles=n_good,
        good_flagged=sum(len(r["false_flags"]) for r in rows),
        good_rejected=sum(len(r["rejected_good"]) for r in rows),
        good_in_crate=sum(len(r["good_in_crate"]) for r in rows),
        bottles_on_floor=sum(len(r["on_floor"]) for r in rows),
        not_inspected=sum(len(r["not_inspected"]) for r in rows),
        duplicate_inspections=sum(r["duplicate_inspections"] for r in rows),
        per_design=per_design(rows),
        episodes_detail=rows,
    )
    print("\nLine success: {line_success}/{episodes}  |  defects flagged {defects_flagged}/{defective_bottles}, "
          "in bin {defects_rejected}/{defective_bottles}  |  good bottles flagged {good_flagged}/{good_bottles}, "
          "rejected {good_rejected}/{good_bottles}, delivered to crate {good_in_crate}/{good_bottles}  |  "
          "bottles on the floor {bottles_on_floor}  |  never inspected {not_inspected}".format(**report))
    for design, d in report["per_design"].items():
        print(f"  {design:10s} bottles {d['bottles']:3d}  defects {d['defective']:2d}  caught {d['caught']:2d}  "
              f"good rejected {d['good_rejected']}  good delivered {d['good_delivered']}/{d['bottles'] - d['defective']}")
    out = args.out or os.path.join(REPO, "inspection_models", f"line_eval_{report['mode']}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("Report:", out)


if __name__ == "__main__":
    main()
