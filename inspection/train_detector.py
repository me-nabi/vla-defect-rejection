"""Train the bottle detector (overhead camera) on the rendered dataset.

    python -m inspection.train_detector --data detection_data --epochs 14

Fine-tunes torchvision's Faster R-CNN with a MobileNetV3-Large FPN backbone (COCO
pre-trained) to find "upright" and "fallen" bottles. Expects detection_data/{train,val}/
from inspection.generate_detection_data. Keeps the checkpoint with the best validation AP
(IoU 0.5, both classes) in inspection_models/bottle_detector.pt.
"""
import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from inspection.vision import LABELS, build_detector


class DetectionDataset(Dataset):
    def __init__(self, root, train):
        with open(os.path.join(root, "annotations.json")) as f:
            self.samples = json.load(f)["samples"]
        self.root = os.path.join(root, "images")
        self.train = train
        self.jitter = v2.ColorJitter(brightness=0.3, contrast=0.25, saturation=0.25, hue=0.03)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        r = self.samples[i]
        img = v2.functional.to_dtype(v2.functional.to_image(Image.open(os.path.join(self.root, r["file"])).convert("RGB")),
                                     torch.float32, scale=True)
        boxes = torch.tensor(r["boxes"], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor(r["labels"], dtype=torch.int64)
        if self.train:
            img = self.jitter(img)
            h, w = img.shape[1:]
            if torch.rand(1) < 0.5:
                img = img.flip(-1)
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if torch.rand(1) < 0.5:
                img = img.flip(-2)
                boxes[:, [1, 3]] = h - boxes[:, [3, 1]]
        return img, {"boxes": boxes, "labels": labels}


def collate(batch):
    return tuple(zip(*batch))


def box_iou(a, b):
    """IoU matrix between (N, 4) and (M, 4) boxes."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter)


def average_precision(preds, gts, iou=0.5, class_agnostic=False):
    """VOC-style AP over a dataset. preds: list of (boxes, scores, labels); gts: list of (boxes, labels)."""
    classes = [None] if class_agnostic else sorted({int(l) for _, ls in gts for l in ls})
    aps = []
    for c in classes:
        scored, n_gt = [], 0
        for (pb, ps, pl), (gb, gl) in zip(preds, gts):
            pm = np.ones(len(pb), bool) if c is None else (pl == c)
            gm = np.ones(len(gb), bool) if c is None else (gl == c)
            pb_c, ps_c, gb_c = pb[pm], ps[pm], gb[gm]
            n_gt += len(gb_c)
            ious = box_iou(pb_c, gb_c)
            taken = np.zeros(len(gb_c), bool)
            for k in np.argsort(-ps_c):
                j = int(np.argmax(ious[k])) if len(gb_c) else -1
                hit = j >= 0 and ious[k, j] >= iou and not taken[j]
                if hit:
                    taken[j] = True
                scored.append((ps_c[k], hit))
        if n_gt == 0:
            continue
        scored.sort(key=lambda t: -t[0])
        hits = np.array([h for _, h in scored], float)
        tp, fp = np.cumsum(hits), np.cumsum(1 - hits)
        recall, precision = tp / n_gt, tp / np.maximum(tp + fp, 1e-9)
        ap = 0.0
        for t in np.linspace(0, 1, 101):
            p = precision[recall >= t]
            ap += (p.max() if len(p) else 0.0) / 101
        aps.append(ap)
    return float(np.mean(aps)) if aps else 0.0


@torch.no_grad()
def predict(net, loader, device):
    net.eval()
    preds, gts = [], []
    for imgs, targets in loader:
        out = net([im.to(device) for im in imgs])
        for o, t in zip(out, targets):
            preds.append((o["boxes"].cpu().numpy(), o["scores"].cpu().numpy(), o["labels"].cpu().numpy()))
            gts.append((t["boxes"].numpy(), t["labels"].numpy()))
    return preds, gts


def detection_metrics(preds, gts, score_threshold=0.6):
    """AP plus precision/recall at the run-time score threshold (class-agnostic, IoU 0.5)."""
    tp = fp = n_gt = 0
    for (pb, ps, _), (gb, _) in zip(preds, gts):
        pb = pb[ps >= score_threshold]
        n_gt += len(gb)
        ious = box_iou(pb, gb)
        taken = np.zeros(len(gb), bool)
        for k in range(len(pb)):
            j = int(np.argmax(ious[k])) if len(gb) else -1
            if j >= 0 and ious[k, j] >= 0.5 and not taken[j]:
                taken[j] = True
                tp += 1
            else:
                fp += 1
    return dict(ap50=average_precision(preds, gts), ap50_any_class=average_precision(preds, gts, class_agnostic=True),
                ap75_any_class=average_precision(preds, gts, iou=0.75, class_agnostic=True),
                precision=tp / max(tp + fp, 1), recall=tp / max(n_gt, 1), boxes=n_gt)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(REPO, "detection_data"))
    p.add_argument("--out", default=os.path.join(REPO, "inspection_models", "bottle_detector.pt"))
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = DetectionDataset(os.path.join(args.data, "train"), train=True)
    val_ds = DetectionDataset(os.path.join(args.data, "val"), train=False)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, collate_fn=collate,
                          drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, 8, num_workers=args.workers, collate_fn=collate, persistent_workers=True)
    print(f"train {len(train_ds)} images, val {len(val_ds)} images, labels {LABELS}, device {device}")

    net = build_detector(len(LABELS)).to(device)
    params = [q for q in net.parameters() if q.requires_grad]
    opt = torch.optim.SGD(params, lr=args.lr, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.1)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0, total, n = time.time(), 0.0, 0
        for imgs, targets in train_dl:
            imgs = [im.to(device) for im in imgs]
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss = sum(net(imgs, targets).values())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            total, n = total + loss.item(), n + 1
        m = detection_metrics(*predict(net, val_dl, device))
        m.update(epoch=epoch, train_loss=total / n)
        history.append(m)
        print(f"epoch {epoch:2d}  loss {m['train_loss']:.3f}  val: AP50 {m['ap50']:.3f}  AP50 (any class) "
              f"{m['ap50_any_class']:.3f}  AP75 {m['ap75_any_class']:.3f}  precision {m['precision']:.3f}  "
              f"recall {m['recall']:.3f}  ({time.time() - t0:.0f}s)", flush=True)
        if m["ap50"] > best:
            best = m["ap50"]
            torch.save({"state_dict": net.state_dict(), "labels": LABELS, "val_metrics": m, "epoch": epoch}, args.out)
    with open(os.path.join(os.path.dirname(args.out), "detector_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val AP50 {best:.3f}; checkpoint {args.out}")


if __name__ == "__main__":
    main()
