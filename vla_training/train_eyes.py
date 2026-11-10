"""Eye test: can SmolVLA's own vision spot bottle defects well enough (>= 95 %)?

    python -m vla_training.train_eyes                         # full run (big GPU)
    python -m vla_training.train_eyes --frozen                # baseline: SmolVLA's eyes untrained
    python -m vla_training.train_eyes --batch-size 8 --grad-accum 4   # smaller GPU, same effective batch

Takes SmolVLA's vision path exactly as the policy uses it (SigLIP image encoder + connector,
64 tokens per 512x512 image, pixels scaled to [-1, 1]) from the checkpoint the full training
will start from, and puts a small defect head on top: attention pooling over the tokens of
both close-up views (pick_front + pick_back), then a linear layer over 9 classes
(none, good and the 7 defect types). The encoder, connector and head are trained together on
the close-up photos from

    python -m inspection.generate_dataset --station pick ...      (pick_data/)

Pass/fail (the gate before building the full single-model policy), measured on the unseen
test set at the threshold chosen on the validation set:
    defects caught (recall)        >= 95 %
    good bottles flagged           <= 2 %
Writes training_output/eyes/<run>/: best.pt (encoder + connector + head), history.json and
report.json (overall, per defect type, per bottle design, confusion matrix, PASS/FAIL).
"""
import argparse
import csv
import glob
import json
import math
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import v2

# none, good and the 7 defect types, in the same order as inspection.model.CLASSES (written out
# here so this script needs only PyTorch and LeRobot, not the simulator).
CLASSES = ("none", "good", "crack", "chip", "missing_cap", "tilted_cap", "low_fill", "particle", "dented")
DEFECTS = [c for c in CLASSES if c not in ("none", "good")]
DEFECT_IDX = [CLASSES.index(c) for c in DEFECTS]
GATE_RECALL, GATE_FALSE_FLAGS = 0.95, 0.02
DEFAULT_CHECKPOINT = "me-nabi/smolvla-libero-100k-final"
IMAGE_SIZE = 512


class PickPhotos(Dataset):
    """(front, back) close-up photos with labels, from one or more generate_dataset folders."""

    def __init__(self, roots, train, limit=None):
        self.rows = []
        for root in roots:
            with open(os.path.join(root, "labels.csv")) as f:
                for r in csv.DictReader(f):
                    r["dir"] = os.path.join(root, "images")
                    self.rows.append(r)
        if limit:
            self.rows = self.rows[:limit]
        self.labels = np.array([CLASSES.index(r["label"]) for r in self.rows])
        aug = [v2.RandomResizedCrop(IMAGE_SIZE, scale=(0.9, 1.0), ratio=(0.95, 1.05), antialias=True),
               v2.ColorJitter(brightness=0.2, contrast=0.15, saturation=0.1, hue=0.02)] if train else \
              [v2.Resize(IMAGE_SIZE, antialias=True)]
        self.tf = v2.Compose([v2.ToImage(), *aug, v2.ToDtype(torch.float32, scale=True)])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        load = lambda fn: self.tf(Image.open(os.path.join(r["dir"], fn)).convert("RGB")) * 2.0 - 1.0   # SmolVLA range
        return load(r["front"]), load(r["back"]), int(self.labels[i])


class EyesNet(nn.Module):
    """SmolVLA's vision encoder + connector, with a defect head over both views' tokens."""

    def __init__(self, checkpoint, n_classes=len(CLASSES)):
        super().__init__()
        from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy
        policy = SmolVLAPolicy.from_pretrained(checkpoint)
        vlm = policy.model.vlm_with_expert.get_vlm_model()
        self.vision, self.connector = vlm.vision_model, vlm.connector
        del policy, vlm                                   # the language model and action expert aren't needed here
        dim = self.connector.modality_projection.proj.out_features if hasattr(self.connector, "modality_projection") \
            else self.connector(torch.zeros(1, 1024, self.vision.config.hidden_size)).shape[-1]
        self.view_embed = nn.Parameter(torch.zeros(2, 1, dim))
        self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
        self.attn = nn.MultiheadAttention(dim, num_heads=8, batch_first=True)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, n_classes))

    def tokens(self, images):
        h = self.vision(pixel_values=images.to(self.vision.dtype)).last_hidden_state
        return self.connector(h)                          # (B, 64, dim), as SmolVLA's embed_image

    def forward(self, front, back):
        b = front.shape[0]
        t = self.tokens(torch.cat([front, back])).float()
        t = torch.cat([t[:b] + self.view_embed[0], t[b:] + self.view_embed[1]], dim=1)
        pooled, _ = self.attn(self.query.expand(b, -1, -1), t, t)
        return self.head(pooled[:, 0])


@torch.no_grad()
def predict(net, loader, device):
    net.eval()
    probs, labels = [], []
    for front, back, y in loader:
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            logits = net(front.to(device, non_blocking=True), back.to(device, non_blocking=True))
        probs.append(F.softmax(logits.float(), dim=1).cpu())
        labels.append(y)
    return torch.cat(probs).numpy(), torch.cat(labels).numpy()


def defect_mask(labels):
    return np.isin(labels, DEFECT_IDX)


def choose_threshold(probs, labels, max_false_flags=GATE_FALSE_FLAGS):
    """Lowest p_defect threshold that keeps good-bottle false flags <= max_false_flags."""
    p = probs[:, DEFECT_IDX].sum(1)
    good = p[labels == CLASSES.index("good")]
    if len(good) == 0:
        return 0.5
    t = float(np.quantile(good, 1 - max_false_flags, method="higher")) + 1e-6
    return min(max(t, 0.05), 0.95)


def metrics(probs, labels, threshold, designs=None):
    p = probs[:, DEFECT_IDX].sum(1)
    flagged = p >= threshold
    is_def, is_good, is_none = defect_mask(labels), labels == CLASSES.index("good"), labels == CLASSES.index("none")
    pred = probs.argmax(1)
    out = dict(
        threshold=threshold,
        defect_recall=float(flagged[is_def].mean()) if is_def.any() else None,
        good_false_flags=float(flagged[is_good].mean()) if is_good.any() else None,
        none_flagged=float(flagged[is_none].mean()) if is_none.any() else None,
        accuracy=float((pred == labels).mean()),
        defect_type_accuracy=float((pred[is_def] == labels[is_def]).mean()) if is_def.any() else None,
        per_class_recall={c: float(flagged[labels == k].mean()) if c in DEFECTS else float((pred[labels == k] == k).mean())
                          for k, c in enumerate(CLASSES) if (labels == k).any()},
        confusion={c: {c2: int(((labels == k) & (pred == k2)).sum()) for k2, c2 in enumerate(CLASSES)}
                   for k, c in enumerate(CLASSES) if (labels == k).any()},
        samples=int(len(labels)),
    )
    if designs is not None:
        designs = np.asarray(designs)
        out["per_design"] = {d: dict(defect_recall=float(flagged[is_def & (designs == d)].mean()) if (is_def & (designs == d)).any() else None,
                                     good_false_flags=float(flagged[is_good & (designs == d)].mean()) if (is_good & (designs == d)).any() else None)
                             for d in sorted(set(designs) - {""})}
    out["passed"] = bool(out["defect_recall"] is not None and out["defect_recall"] >= GATE_RECALL
                         and out["good_false_flags"] is not None and out["good_false_flags"] <= GATE_FALSE_FLAGS)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", nargs="+", default=sorted(glob.glob(os.path.join(REPO, "pick_data", "train_*"))))
    p.add_argument("--val", default=os.path.join(REPO, "pick_data", "val"))
    p.add_argument("--test", default=os.path.join(REPO, "pick_data", "test"))
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT, help="SmolVLA checkpoint (HF id or folder)")
    p.add_argument("--out", default=os.path.join(REPO, "training_output", "eyes"))
    p.add_argument("--run", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr-vision", type=float, default=2e-5)
    p.add_argument("--lr-connector", type=float, default=1e-4)
    p.add_argument("--lr-head", type=float, default=1e-3)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--frozen", action="store_true", help="baseline: keep SmolVLA's encoder and connector frozen, train the head only")
    p.add_argument("--limit", type=int, default=None, help="use only this many samples per split (smoke test)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = args.run or ("frozen" if args.frozen else "finetuned")
    out_dir = os.path.join(args.out, run)
    os.makedirs(out_dir, exist_ok=True)
    if not args.train:
        raise SystemExit("No training folders found (pick_data/train_*). Generate them first.")

    train_ds = PickPhotos(args.train, train=True, limit=args.limit)
    val_ds = PickPhotos([args.val], train=False, limit=args.limit)
    kw = dict(num_workers=args.workers, pin_memory=device == "cuda", persistent_workers=args.workers > 0)
    train_dl = DataLoader(train_ds, args.batch_size, shuffle=True, drop_last=True, **kw)
    val_dl = DataLoader(val_ds, args.batch_size, **kw)
    print(f"train {len(train_ds)} pairs from {len(args.train)} folder(s), val {len(val_ds)}, device {device}", flush=True)

    net = EyesNet(args.checkpoint).to(device)
    head_params = [*net.attn.parameters(), *net.head.parameters(), net.query, net.view_embed]
    groups = [dict(params=head_params, lr=args.lr_head)]
    for part, lr in ((net.vision, args.lr_vision), (net.connector, args.lr_connector)):
        part.requires_grad_(not args.frozen)
        if not args.frozen:
            groups.append(dict(params=list(part.parameters()), lr=lr))
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    steps = args.epochs * max(len(train_dl) // args.grad_accum, 1)
    warmup = max(int(0.05 * steps), 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s / steps, 1.0))))
    n_train = sum(q.numel() for g in groups for q in g["params"])
    print(f"trainable parameters: {n_train / 1e6:.1f} M ({'head only' if args.frozen else 'encoder + connector + head'}),"
          f" {steps} optimizer steps", flush=True)

    best, history = -1.0, []
    for epoch in range(1, args.epochs + 1):
        net.train()
        if args.frozen:
            net.vision.eval(), net.connector.eval()
        t0, total, n = time.time(), 0.0, 0
        opt.zero_grad(set_to_none=True)
        for i, (front, back, y) in enumerate(train_dl):
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
                logits = net(front.to(device, non_blocking=True), back.to(device, non_blocking=True))
                loss = F.cross_entropy(logits.float(), y.to(device), label_smoothing=0.05)
            (loss / args.grad_accum).backward()
            if (i + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_([q for g in groups for q in g["params"]], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
            total, n = total + loss.item(), n + 1
            if n % 200 == 0:
                print(f"  epoch {epoch} step {n}/{len(train_dl)} loss {total / n:.4f}", flush=True)
        probs, labels = predict(net, val_dl, device)
        th = choose_threshold(probs, labels)
        m = metrics(probs, labels, th)
        m.update(epoch=epoch, train_loss=total / max(n, 1), minutes=(time.time() - t0) / 60)
        history.append(m)
        print(f"epoch {epoch:2d}  loss {m['train_loss']:.4f}  val: defects caught {m['defect_recall']:.3f} "
              f"at {m['good_false_flags']:.3f} false flags (threshold {th:.2f}), type accuracy "
              f"{m['defect_type_accuracy']:.3f}, accuracy {m['accuracy']:.3f}  ({m['minutes']:.1f} min)", flush=True)
        if m["defect_recall"] > best:
            best = m["defect_recall"]
            torch.save(dict(state_dict=net.state_dict(), classes=CLASSES, threshold=th, epoch=epoch,
                            checkpoint=args.checkpoint, frozen=args.frozen, val=m), os.path.join(out_dir, "best.pt"))
        with open(os.path.join(out_dir, "history.json"), "w") as f:
            json.dump(history, f, indent=1)

    # final verdict on the unseen test set, with the best checkpoint and its validation threshold
    ck = torch.load(os.path.join(out_dir, "best.pt"), map_location=device, weights_only=False)
    net.load_state_dict(ck["state_dict"])
    test_ds = PickPhotos([args.test], train=False, limit=args.limit)
    probs, labels = predict(net, DataLoader(test_ds, args.batch_size, **kw), device)
    report = dict(run=run, checkpoint=args.checkpoint, best_epoch=ck["epoch"],
                  gate=dict(defect_recall=GATE_RECALL, good_false_flags=GATE_FALSE_FLAGS),
                  test=metrics(probs, labels, ck["threshold"], designs=[r["design"] for r in test_ds.rows]))
    with open(os.path.join(out_dir, "report.json"), "w") as f:
        json.dump(report, f, indent=1)
    t = report["test"]
    print(f"\nTEST ({t['samples']} pairs): defects caught {t['defect_recall']:.3f}, good bottles flagged "
          f"{t['good_false_flags']:.3f}, defect type accuracy {t['defect_type_accuracy']:.3f}")
    print("per class:", {k: round(v, 3) for k, v in t["per_class_recall"].items()})
    print("RESULT:", "PASS - SmolVLA's eyes are good enough" if t["passed"] else "FAIL - below the 95 % gate")
    print("report:", os.path.join(out_dir, "report.json"))


if __name__ == "__main__":
    main()
