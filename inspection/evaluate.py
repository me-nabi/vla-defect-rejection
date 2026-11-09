"""Test-set metrics for the inspection classifier.

    python -m inspection.evaluate --data inspection_data/test

Prints and saves (inspection_models/test_metrics.json):
  - good vs defective at the checkpoint's threshold: accuracy, defect recall, false-flag rate
  - recall per defect type (share of that type flagged as defective)
  - the 7-way confusion matrix (rows = truth, columns = prediction)
and writes example and error galleries to docs/media/.
"""
import argparse
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.utils.data import DataLoader

from inspection.model import CLASSES, InspectionNet
from inspection.train import PairDataset, binary_metrics, predict


def gallery(ds, idx, title_fn, path, cols=4, thumb=160):
    tiles = []
    for i in idx:
        r = ds.rows[i]
        pair = [Image.open(os.path.join(ds.root, r[k])).convert("RGB").resize((thumb, thumb)) for k in ("front", "back")]
        tile = Image.new("RGB", (2 * thumb, thumb + 30), (15, 15, 20))
        tile.paste(pair[0], (0, 30))
        tile.paste(pair[1], (thumb, 30))
        ImageDraw.Draw(tile).text((4, 2), title_fn(i), fill=(240, 240, 240))
        tiles.append(np.array(tile))
    if not tiles:
        return
    while len(tiles) % cols:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.concatenate(tiles[k:k + cols], 1) for k in range(0, len(tiles), cols)]
    Image.fromarray(np.concatenate(rows, 0)).save(path)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(REPO, "inspection_data", "test"))
    p.add_argument("--model", default=os.path.join(REPO, "inspection_models", "inspection_net.pt"))
    p.add_argument("--media", default=os.path.join(REPO, "docs", "media"))
    p.add_argument("--tag", default="test", help="name used for the metrics file and galleries")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.model, map_location="cpu", weights_only=False)
    net = InspectionNet(len(CLASSES), pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    thr = float(ckpt["threshold"])

    ds = PairDataset(args.data, train=False)
    p_defect, preds, labels = predict(net, DataLoader(ds, 64, num_workers=6), device)
    m = binary_metrics(p_defect, labels, thr)
    flagged = p_defect >= thr
    per_type = {c: float(flagged[labels == k].mean()) for k, c in enumerate(CLASSES) if k > 0 and (labels == k).any()}
    conf = np.zeros((len(CLASSES), len(CLASSES)), int)
    for t, q in zip(labels, preds):
        conf[t, q] += 1
    designs = np.array([r.get("design", "?") for r in ds.rows])
    per_design = {}
    for d in sorted(set(designs)):
        k = designs == d
        good, bad = k & (labels == 0), k & (labels > 0)
        per_design[d] = {"pairs": int(k.sum()), "accuracy": float((flagged[k] == (labels[k] > 0)).mean()),
                         "defect_recall": float(flagged[bad].mean()) if bad.any() else None,
                         "false_flag_rate": float(flagged[good].mean()) if good.any() else None}
    report = {"test_pairs": len(ds), "threshold": thr, **m, "type_accuracy": float((preds == labels).mean()),
              "recall_per_defect": per_type, "per_design": per_design, "confusion_matrix": conf.tolist(),
              "classes": CLASSES, "checkpoint_epoch": ckpt.get("epoch")}

    print(f"Test pairs: {len(ds)}   threshold {thr:.2f}")
    print(f"good vs defective: accuracy {m['accuracy']:.3f}  defect recall {m['defect_recall']:.3f}  "
          f"false-flag rate {m['false_flag_rate']:.3f}")
    print("recall per defect type:", {k: round(v, 3) for k, v in per_type.items()})
    for d, v in per_design.items():
        print(f"  {d:10s} pairs {v['pairs']:5d}  accuracy {v['accuracy']:.3f}  defect recall {v['defect_recall'] or 0:.3f}  "
              f"false flags {v['false_flag_rate'] or 0:.3f}")
    print("confusion (rows truth, cols prediction):", " ".join(f"{c[:6]:>7}" for c in CLASSES))
    for c, row in zip(CLASSES, conf):
        print(f"  {c:12s}" + " ".join(f"{v:7d}" for v in row))

    out = os.path.join(os.path.dirname(args.model), f"{args.tag}_metrics.json")
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print("Metrics:", out)

    os.makedirs(args.media, exist_ok=True)
    rng = np.random.default_rng(0)
    examples = [int(rng.choice(np.flatnonzero(labels == k))) for k in range(len(CLASSES)) if (labels == k).any()]
    def title(i):
        verdict = "OK"
        if flagged[i]:
            verdict = "DEFECT " + (CLASSES[preds[i]] if preds[i] != 0 else "(type uncertain)")
        return f"truth: {CLASSES[labels[i]]}  ->  {verdict} (p={p_defect[i]:.2f})"
    gallery(ds, examples, title, os.path.join(args.media, f"inspection_{args.tag}_examples.png"))
    errors = np.flatnonzero(flagged != (labels != 0))[:12]
    gallery(ds, errors, title, os.path.join(args.media, f"inspection_{args.tag}_errors.png"))
    print(f"Galleries: {args.media}/inspection_{args.tag}_examples.png, inspection_{args.tag}_errors.png "
          f"({len(errors)} errors shown)")


if __name__ == "__main__":
    main()
