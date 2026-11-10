"""Inspection station: photograph each bottle as it passes and decide whether it's defective.

The station works on the tracks of the vision system (inspection/vision.py): a tracked
bottle is photographed the moment its track reaches the station (env.INSPECTION_X), and the
verdict is attached to that track. The station never sees the simulator's object names.

Ground truth is used in exactly two places, both outside the robot's decisions:
`--oracle` mode (decisions from the simulator, to test the robot on its own) and scoring
(`summary`), which matches each inspected track to the real bottle nearest to it.
"""
import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from PIL import Image

from inspection.model import IMAGE_SIZE, REASONS


@dataclass
class Verdict:
    defective: bool
    p_defect: float              # model confidence that the bottle is defective (1.0/0.0 for the oracle)
    defect: Optional[str]        # predicted defect type, or None
    source: str                  # "model" or "oracle"
    heatmaps: Optional[tuple] = None   # Grad-CAM (front, back) for defective verdicts from the model

    @property
    def reason(self):
        """Why the bottle is rejected, in plain words ("" for good bottles)."""
        return REASONS.get(self.defect, "Defect") if self.defective else ""

    @property
    def confidence(self):
        return self.p_defect if self.defective else 1.0 - self.p_defect


# Simple industrial-camera model: sensor noise, then JPEG compression on the way to the PC.
# The dataset generator adds the same noise and saves JPEGs at the same quality, so the
# classifier sees identical image statistics in training and on the line.
SENSOR_NOISE = 2.5      # std of additive Gaussian noise, in 8-bit grey levels
JPEG_QUALITY = 92


def render_views(env, size=IMAGE_SIZE, cameras=None):
    """(front, back) clean uint8 renders from the two inspection cameras (or `cameras`), upright."""
    return tuple(env.sim.render(width=size, height=size, camera_name=cam)[::-1].copy()
                 for cam in (cameras or env.INSPECTION_CAMERAS))


def add_sensor_noise(img, rng, sigma=SENSOR_NOISE):
    return np.clip(img.astype(np.float32) + rng.normal(0, sigma, img.shape), 0, 255).astype(np.uint8)


def jpeg_roundtrip(img):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, "JPEG", quality=JPEG_QUALITY)
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def capture(env, rng=None):
    """(front, back) as the station's cameras deliver them: rendered, noisy, JPEG-compressed."""
    rng = np.random.default_rng() if rng is None else rng
    return tuple(jpeg_roundtrip(add_sensor_noise(v, rng)) for v in render_views(env))


def nearest_bottle(env, xy):
    """Simulator bottle closest to a world xy. Ground truth: only for scoring and --oracle."""
    names = env.inspection_bottles
    d = [np.linalg.norm(env.sim.data.body_xpos[env.obj_body_id[n]][:2] - xy) for n in names]
    return names[int(np.argmin(d))], float(min(d))


class InspectionStation:
    """Inspects tracked bottles as they pass. classifier=None uses the simulator's ground truth."""

    def __init__(self, env, classifier=None):
        self.env = env
        self.classifier = classifier
        self.rng = np.random.default_rng(np.random.randint(2**31))   # camera noise
        self.verdicts = {}       # track id -> Verdict
        self.photos = {}         # track id -> (front, back) inspection photos
        self.last = None         # (track label, front, back, verdict) of the latest inspection, for the HUD
        self.inspections = []    # one record per inspection; "truth" is filled in for scoring only

    @property
    def oracle(self):
        return self.classifier is None

    def update(self, tracks):
        """Inspect tracks that reached the station this step. Returns those tracks.

        A track is only inspected if it was first seen upstream of the station; a bottle that
        is first picked up downstream (it was never photographed) stays uninspected."""
        new = []
        x_station = self.env.INSPECTION_X
        for t in tracks:
            if t.id in self.verdicts or t.xy[0] > x_station or t.born_x < x_station + 0.02 or not t.upright:
                continue
            front, back = capture(self.env, self.rng)
            truth, dist = nearest_bottle(self.env, t.xy)          # scoring / oracle only
            if self.classifier is None:
                defect = self.env.bottle_defects[truth]
                verdict = Verdict(defect is not None, float(defect is not None), defect, "oracle")
            else:
                verdict = Verdict(*self.classifier(front, back), "model")
                if verdict.defective and hasattr(self.classifier, "explain"):
                    verdict.heatmaps = self.classifier.explain(front, back, verdict.defect)
            t.verdict = verdict
            self.verdicts[t.id] = verdict
            self.photos[t.id] = (front, back)
            self.last = (t.label, front, back, verdict)
            self.inspections.append(dict(track=t.id, xy=t.xy.copy(), verdict=verdict, truth=truth, truth_dist=dist))
            new.append(t)
        return new

    def flagged(self):
        """Track ids judged defective."""
        return sorted(i for i, v in self.verdicts.items() if v.defective)

    def summary(self, rejected):
        """Compare decisions with ground truth. `rejected` = bottles (simulator names) that
        ended up in the bin; each inspection is matched to the bottle nearest its track."""
        truth = self.env.bottle_defects
        bad = {n for n, d in truth.items() if d is not None}
        inspected = {i["truth"] for i in self.inspections}
        flagged = {i["truth"] for i in self.inspections if i["verdict"].defective}
        rejected = set(rejected)
        return {
            "defective": len(bad),
            "flagged": len(flagged),
            "correctly_flagged": len(bad & flagged),
            "missed_defects": sorted(bad - flagged),
            "false_flags": sorted(flagged - bad),
            "not_inspected": sorted(set(truth) - inspected),
            "duplicate_inspections": len(self.inspections) - len(inspected),
            "rejected_defects": len(bad & rejected),
            "rejected_good": sorted(rejected - bad),
        }


def downstream_first(tracks):
    """Order tracks by position along the belt, most downstream (closest to the robot) first."""
    return sorted(tracks, key=lambda t: t.xy[0])
