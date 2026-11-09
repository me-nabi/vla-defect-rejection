"""Finding and following bottles with our own vision model: an overhead RGB-D camera, a
trained detector and a tracker. Nothing here reads the simulator's object list or
positions; the robot learns where bottles are only from the camera.

    camera  -> OverheadCamera: colour image + depth (with sensor noise), calibrated pose
    detect  -> BottleDetector: Faster R-CNN (MobileNetV3-FPN) fine-tuned on simulated images,
               boxes for "upright" and "fallen" bottles
    locate  -> localize(): 3D position from the depth pixels inside a box. For an upright
               bottle the cap is the highest surface, so the cap pixels give its axis and the
               top height gives its height (which tells the robot where to grasp)
    track   -> Tracker: keeps an ID per bottle between frames. Between detections it moves
               tracks with the conveyor (a belt encoder: the line knows its own belt speed)

The only things taken from the cell setup are calibration values: the camera pose and
intrinsics, the belt surface height and where the reject box stands.
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

LABELS = ("background", "upright", "fallen")
BELT_Z = 0.90               # belt surface height (cell calibration)
DEPTH_NOISE = 0.0015        # m, std of the simulated depth sensor noise
VIEW_X_MIN = -0.17          # tracks leaving the camera's view downstream are dropped here


# ---- camera ---------------------------------------------------------------------------
class OverheadCamera:
    """The overhead RGB-D camera (env.TRACK_CAMERA) with a simple sensor model."""

    def __init__(self, env, rng=None):
        self.env = env
        self.name = env.TRACK_CAMERA
        self.h, self.w = env.TRACK_CAM_SIZE
        self.rng = np.random.default_rng() if rng is None else rng

    def capture(self):
        """-> (rgb uint8 HxWx3, depth float32 HxW in metres along the optical axis), upright."""
        from robosuite.utils.camera_utils import get_real_depth_map
        from inspection.station import add_sensor_noise, jpeg_roundtrip
        sim = self.env.sim
        rgb, depth = sim.render(width=self.w, height=self.h, camera_name=self.name, depth=True)
        rgb = jpeg_roundtrip(add_sensor_noise(rgb[::-1], self.rng))
        depth = get_real_depth_map(sim, depth[::-1]).astype(np.float32)
        depth = depth + self.rng.normal(0, DEPTH_NOISE, depth.shape).astype(np.float32)
        return rgb, depth

    def unproject(self, us, vs, depth):
        """Pixel coordinates (upright image) + depths -> world points (N, 3)."""
        m, d = self.env.sim.model, self.env.sim.data
        cid = m.camera_name2id(self.name)
        f = (self.h / 2.0) / np.tan(np.radians(m.cam_fovy[cid]) / 2.0)
        xc = (us + 0.5 - self.w / 2.0) / f * depth
        yc = -(vs + 0.5 - self.h / 2.0) / f * depth
        pts_cam = np.stack([xc, yc, -depth], axis=1)
        return d.cam_xpos[cid] + pts_cam @ d.cam_xmat[cid].reshape(3, 3).T


# ---- detector -------------------------------------------------------------------------
def build_detector(num_classes=len(LABELS), pretrained=True):
    import torchvision
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    weights = "DEFAULT" if pretrained else None
    net = torchvision.models.detection.fasterrcnn_mobilenet_v3_large_fpn(
        weights=weights, weights_backbone="DEFAULT" if pretrained else None, min_size=384, max_size=512)
    in_features = net.roi_heads.box_predictor.cls_score.in_features
    net.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return net


class BottleDetector:
    """Loads a trained detector checkpoint; call it on an RGB image."""

    def __init__(self, path, device=None, score_threshold=0.6):
        import torch
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        self.labels = tuple(ckpt["labels"])
        self.net = build_detector(len(self.labels), pretrained=False)
        self.net.load_state_dict(ckpt["state_dict"])
        self.net.to(self.device).eval()
        self.score_threshold = score_threshold

    def __call__(self, rgb):
        """-> list of (box [x1, y1, x2, y2] in pixels, score, label name)."""
        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div(255).to(self.device)
            out = self.net([x])[0]
        keep = out["scores"] >= self.score_threshold
        return [(b, float(s), self.labels[int(l)]) for b, s, l in
                zip(out["boxes"][keep].cpu().numpy(), out["scores"][keep].cpu().numpy(), out["labels"][keep].cpu().numpy())]


# ---- 3D localisation ------------------------------------------------------------------
@dataclass
class Detection:
    box: np.ndarray            # pixels in the overhead image
    score: float
    upright: bool
    xy: np.ndarray             # bottle axis (upright) or body centre (fallen), world metres
    top_z: float               # highest point of the bottle
    radius: float
    bounds: np.ndarray         # (2, 3) world-space box around the bottle's visible points

    @property
    def height(self):
        return self.top_z - BELT_Z


def localize(camera, depth, box, score, label, min_points=15):
    """3D position of a detected bottle from the depth pixels inside its box."""
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    x1, y1 = max(x1, 0), max(y1, 0)
    x2, y2 = min(x2, camera.w), min(y2, camera.h)
    if x2 - x1 < 3 or y2 - y1 < 3:
        return None
    vs, us = np.mgrid[y1:y2, x1:x2]
    pts = camera.unproject(us.ravel().astype(float), vs.ravel().astype(float), depth[y1:y2, x1:x2].ravel())
    pts = pts[(pts[:, 2] > BELT_Z + 0.008) & (pts[:, 2] < BELT_Z + 0.4)]     # the bottle, not the belt
    if len(pts) < min_points:
        return None
    top_z = float(np.percentile(pts[:, 2], 99))
    upright = label == "upright"
    if upright:
        cap = pts[pts[:, 2] > top_z - 0.012]
        xy = cap[:, :2].mean(axis=0)
        body = pts[pts[:, 2] > BELT_Z + 0.02]
        if len(body) < min_points:           # nothing standing up: not a bottle
            return None
        radius = float(np.clip(np.percentile(np.linalg.norm(body[:, :2] - xy, axis=1), 95), 0.015, 0.04))
        lo = np.r_[xy - radius, BELT_Z]
        hi = np.r_[xy + radius, top_z]
    else:
        xy = pts[:, :2].mean(axis=0)
        radius = float(np.clip((top_z - BELT_Z) / 2, 0.015, 0.04))
        lo = np.r_[pts[:, :2].min(axis=0), BELT_Z]
        hi = np.r_[pts[:, :2].max(axis=0), top_z]
    return Detection(np.asarray(box, float), score, upright, xy, top_z, radius, np.stack([lo, hi]))


# ---- tracking -------------------------------------------------------------------------
@dataclass
class Track:
    id: int
    xy: np.ndarray
    top_z: float
    upright: bool
    radius: float
    bounds: np.ndarray
    box: np.ndarray
    born_x: float
    hits: int = 1
    misses: int = 0
    verdict: Optional[object] = None          # set by the inspection station
    handled: bool = False                     # the robot has dealt with it
    held: bool = False                        # in the gripper: follows the gripper, not the belt
    history: list = field(default_factory=list)

    @property
    def height(self):
        return self.top_z - BELT_Z

    @property
    def label(self):
        return f"#{self.id}"

    def follow_gripper(self, eef, hold_dz):
        """While carried: the bottle hangs hold_dz below the gripper."""
        base = np.r_[eef[:2], eef[2] - hold_dz]
        self.xy = base[:2].copy()
        self.bounds = np.stack([base - np.r_[self.radius, self.radius, 0], base + np.r_[self.radius, self.radius, self.height]])

    def absorb(self, det):
        self.xy, self.top_z, self.upright = det.xy.copy(), det.top_z, det.upright
        self.radius, self.bounds, self.box = det.radius, det.bounds.copy(), det.box
        self.hits += 1
        self.misses = 0


class Tracker:
    """Nearest-neighbour tracker on the belt plane, with conveyor-motion prediction.

    exclude: list of (centre_xy, half_size_xy) areas where detections are ignored (the reject
    box: bottles already rejected stay in view there)."""

    def __init__(self, gate=0.035, max_misses=10, new_track_score=0.8, exclude=(), revive_gate=0.05, memory=80):
        self.gate, self.max_misses, self.new_track_score = gate, max_misses, new_track_score
        self.revive_gate, self.memory = revive_gate, memory
        self.exclude = [(np.asarray(c, float), np.asarray(h, float)) for c, h in exclude]
        self.tracks = []
        self.lost = []            # recently lost tracks, still moved with the belt, for re-identification
        self._next_id = 1

    def _excluded(self, xy):
        return any(np.all(np.abs(xy - c) <= h) for c, h in self.exclude)

    def predict(self, belt_velocity, dt):
        """Move every track with the belt. belt_velocity: function (x, y) -> (vx, vy)."""
        for t in self.tracks + self.lost:
            if t.held:
                continue
            v = np.asarray(belt_velocity(*t.xy)) * dt
            t.xy = t.xy + v
            t.bounds = t.bounds + np.r_[v, 0.0]
        for t in self.lost:
            t.misses += 1
        self.tracks = [t for t in self.tracks if t.xy[0] > VIEW_X_MIN]
        self.lost = [t for t in self.lost if t.xy[0] > VIEW_X_MIN and t.misses <= self.memory]

    def update(self, detections):
        dets = [d for d in detections if not self._excluded(d.xy) and abs(d.xy[1]) < 0.2
                and BELT_Z + 0.03 < d.top_z < BELT_Z + 0.22]
        pairs = sorted(((np.linalg.norm(t.xy - d.xy), i, j) for i, t in enumerate(self.tracks) if not t.held
                        for j, d in enumerate(dets)), key=lambda p: p[0])
        used_t, used_d = set(), set()
        for dist, i, j in pairs:
            if dist > self.gate or i in used_t or j in used_d:
                continue
            self.tracks[i].absorb(dets[j])
            used_t.add(i)
            used_d.add(j)
        for i, t in enumerate(self.tracks):
            if i not in used_t and not t.held:
                t.misses += 1
        for j, d in enumerate(dets):
            if j in used_d or d.score < self.new_track_score:
                continue
            # same bottle seen again after a gap (e.g. the arm passed over it)?
            near = sorted(self.lost, key=lambda t: np.linalg.norm(t.xy - d.xy))
            if near and np.linalg.norm(near[0].xy - d.xy) < self.revive_gate:
                t = near[0]
                self.lost.remove(t)
                t.absorb(d)
                self.tracks.append(t)
                continue
            self.tracks.append(Track(self._next_id, d.xy.copy(), d.top_z, d.upright, d.radius,
                                     d.bounds.copy(), d.box, born_x=float(d.xy[0])))
            self._next_id += 1
        dropped = [t for t in self.tracks if t.misses > self.max_misses]
        self.lost += [t for t in dropped if not t.handled]
        self.tracks = [t for t in self.tracks if t.misses <= self.max_misses]


class LineVision:
    """Camera + detector + tracker, stepped once per control step by the line controller."""

    def __init__(self, env, detector, detect_every=2, exclude=(), rng=None, dt=0.05):
        self.env = env
        self.camera = OverheadCamera(env, rng)
        self.detector = detector
        self.tracker = Tracker(exclude=exclude)
        self.detect_every = detect_every
        self.dt = dt
        self.last_rgb, self.last_detections = None, []

    @property
    def tracks(self):
        return self.tracker.tracks

    def belt_velocity(self, x, y):
        """The main belt's velocity: what the line's own encoder reports (0 while paused)."""
        return (-self.env.belt_speed, 0.0) if self.env.belt_active else (0.0, 0.0)

    def step(self, step, force=False):
        self.tracker.predict(self.belt_velocity, self.dt)
        if force or step % self.detect_every == 0:
            self.detect()

    def detect(self):
        rgb, depth = self.camera.capture()
        dets = [localize(self.camera, depth, box, score, label) for box, score, label in self.detector(rgb)]
        dets = [d for d in dets if d is not None]
        self.last_rgb, self.last_detections = rgb, dets
        self.tracker.update(dets)
        return dets

    def track(self, track_id):
        return next((t for t in self.tracks if t.id == track_id), None)
