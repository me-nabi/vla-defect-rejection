"""Train the bottle inspection classifier on the rendered dataset.

    python -m inspection.train --data inspection_data --epochs 12

Expects inspection_data/{train,val}/labels.csv from inspection.generate_dataset. Saves the
checkpoint with the best validation balanced accuracy (good vs defective) to
inspection_models/inspection_net.pt, together with the decision threshold chosen on the
validation set.
"""
import argparse
import csv
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

from inspection.model import CLASSES, IMAGE_SIZE, InspectionNet

MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class PairDataset(Dataset):
    def __init__(self, root, train):
        with open(os.path.join(root, "labels.csv")) as f:
            self.rows = list(csv.DictReader(f))
        self.root = os.path.join(root, "images")
        self.labels = np.array([CLASSES.index(r["label"]) for r in self.rows])
        aug = [v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.88, 1.0), ratio=(0.95, 1.05), antialias=True),
               v2.RandomHorizontalFlip(),
               v2.ColorJitter(brightness=0.25, contrast=0.2, saturation=0.15, hue=0.02)] if train else []
        self.tf = v2.Compose([v2.ToImage(), *aug, v2.ToDtype(torch.float32, scale=True), v2.Normalize(MEAN, STD)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        load = lambda fn: self.tf(Image.open(os.path.join(self.root, fn)).convert("RGB"))
        return load(r["front"]), load(r["back"]), int(self.labels[i])


def defect_probability(logits):
    return 1.0 - torch.softmax(logits.float(), dim=1)[:, CLASSES.index("good")]


@torch.inference_mode()
def predict(net, loader, device):
    net.eval()
    probs, labels, preds = [], [], []
    for front, back, y in loader:
        with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
            logits = net(front.to(device, non_blocking=True), back.to(device, non_blocking=True))
        probs.append(defect_probability(logits).cpu())
        preds.append(logits.argmax(1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(preds).numpy(), torch.cat(labels).numpy()


def binary_metrics(p_defect, labels, threshold):
    is_bad = labels != CLASSES.index("good")
    flagged = p_defect >= threshold
    recall = float((flagged & is_bad).sum() / max(is_bad.sum(), 1))           # defects caught
    false_flag = float((flagged & ~is_bad).sum() / max((~is_bad).sum(), 1))   # good bottles flagged
    return {"accuracy": float((flagged == is_bad).mean()), "defect_recall": recall,
            "false_flag_rate": false_flag, "balanced_accuracy": (recall + 1 - false_flag) / 2}


def best_threshold(p_defect, labels):
    """Centre of the range of thresholds with the best validation balanced accuracy.

    On clean validation data a wide range of thresholds is equally good; taking the
    middle of it (instead of its lowest point) leaves the most margin when conditions
    drift, e.g. dimmer light or more sensor noise, where the model gets less confident.
    """
    grid = np.linspace(0.05, 0.95, 91)
    scores = np.array([binary_metrics(p_defect, labels, t)["balanced_accuracy"] for t in grid])
    best = np.flatnonzero(scores >= scores.max() - 1e-9)
    return float(np.median(grid[best]))


def recalibrate(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.out, map_location="cpu", weights_only=False)
    net = InspectionNet(len(CLASSES), pretrained=False)
    net.load_state_dict(ckpt["state_dict"])
    net.to(device)
    val_dl = DataLoader(PairDataset(os.path.join(args.data, "val"), train=False), 64, num_workers=args.workers)
    p_defect, _, labels = predict(net, val_dl, device)
    old, ckpt["threshold"] = ckpt["threshold"], best_threshold(p_defect, labels)
    ckpt["val_metrics"].update(binary_metrics(p_defect, labels, ckpt["threshold"]), threshold=ckpt["threshold"])
    torch.save(ckpt, args.out)
    print(f"threshold {old:.2f} -> {ckpt['threshold']:.2f} (validation: {ckpt['val_metrics']})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=os.path.join(REPO, "inspection_data"))
    p.add_argument("--out", default=os.path.join(REPO, "inspection_models", "inspection_net.pt"))
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--recalibrate", action="store_true",
                   help="don't train: recompute the decision threshold of --out on the validation set")
    args = p.parse_args()
    if args.recalibrate:
        return recalibrate(args)

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = PairDataset(os.path.join(args.data, "train"), train=True)
    val_ds = PairDataset(os.path.join(args.data, "val"), train=False)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True,
                          drop_last=True, persistent_workers=True)
    val_dl = DataLoader(val_ds, 64, num_workers=args.workers, pin_memory=True, persistent_workers=True)
    print(f"train {len(train_ds)} pairs, val {len(val_ds)} pairs, classes {CLASSES}, device {device}")

    net = InspectionNet(len(CLASSES)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(train_dl), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device == "cuda")
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.05)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        net.train()
        t0, total, n = time.time(), 0.0, 0
        for front, back, y in train_dl:
            front, back, y = (t.to(device, non_blocking=True) for t in (front, back, y))
            with torch.autocast(device_type=device, dtype=torch.float16, enabled=device == "cuda"):
                loss = loss_fn(net(front, back), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total, n = total + loss.item() * len(y), n + len(y)

        p_defect, preds, labels = predict(net, val_dl, device)
        thr = best_threshold(p_defect, labels)
        m = binary_metrics(p_defect, labels, thr)
        m.update(epoch=epoch, train_loss=total / n, type_accuracy=float((preds == labels).mean()), threshold=thr)
        history.append(m)
        print(f"epoch {epoch:2d}  loss {m['train_loss']:.3f}  val: balanced acc {m['balanced_accuracy']:.3f}  "
              f"defect recall {m['defect_recall']:.3f}  false flags {m['false_flag_rate']:.3f}  "
              f"7-way acc {m['type_accuracy']:.3f}  thr {thr:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        if m["balanced_accuracy"] > best:
            best = m["balanced_accuracy"]
            torch.save({"state_dict": net.state_dict(), "classes": CLASSES, "threshold": thr,
                        "image_size": IMAGE_SIZE, "val_metrics": m, "epoch": epoch}, args.out)

    with open(os.path.join(os.path.dirname(args.out), "train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"best val balanced accuracy {best:.3f}; checkpoint {args.out}")


if __name__ == "__main__":
    main()
