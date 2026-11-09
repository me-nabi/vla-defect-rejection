"""Video overlay for the conveyor line: per-bottle labels, HUD, inspection panel and the live
dashboard window.

Everything drawn comes from the vision system and the inspection station: boxes are the
tracker's 3D estimates (from the overhead RGB-D camera) projected into each view, IDs are
the tracker's own (#1, #2, ...), and verdicts are the model's (or "ground truth" in oracle
mode). Nothing is drawn from the simulator's object positions.
"""
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GREY, GREEN, RED, CYAN = (150, 150, 150), (50, 210, 50), (235, 50, 50), (60, 200, 230)
_FONT_FILES = {False: ("DejaVuSans.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
               True: ("DejaVuSans-Bold.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")}


@lru_cache(maxsize=None)
def font(size, bold=False):
    """Scalable TrueType font (DejaVu if installed, else Pillow's built-in one)."""
    for f in _FONT_FILES[bold]:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    return ImageFont.load_default(size=size)


def project(sim, points, cam_name, img_h, img_w):
    """World points -> (N, 2) pixel coordinates (unclipped); None if any point is behind the camera."""
    cam_id = sim.model.camera_name2id(cam_name)
    cam_mat = sim.data.cam_xmat[cam_id].reshape(3, 3)
    c = (np.asarray(points) - sim.data.cam_xpos[cam_id]) @ cam_mat
    if np.any(c[:, 2] >= -0.01):
        return None
    f = img_h / (2.0 * np.tan(np.radians(sim.model.cam_fovy[cam_id]) / 2.0))
    return np.stack([img_w / 2.0 + f * c[:, 0] / -c[:, 2], img_h / 2.0 - f * c[:, 1] / -c[:, 2]], axis=1)


def track_box(env, track, cam_name, img_h, img_w, margin=3):
    """Pixel rectangle of a track's measured 3D box in a camera view, or None if off-screen."""
    lo, hi = track.bounds
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    pts = project(env.sim, corners, cam_name, img_h, img_w)
    if pts is None:
        return None
    x1, y1 = pts.min(axis=0) - margin
    x2, y2 = pts.max(axis=0) + margin
    if x2 < 0 or y2 < 0 or x1 >= img_w or y1 >= img_h:
        return None
    return int(max(0, x1)), int(max(0, y1)), int(min(img_w - 1, x2)), int(min(img_h - 1, y2))


def short_label(verdict):
    """Compact tag drawn over each bottle in the line view."""
    if verdict is None:
        return "?", GREY
    return ("REJECT", RED) if verdict.defective else ("OK", GREEN)


def confidence_text(verdict):
    return "ground truth" if verdict.source == "oracle" else f"{verdict.confidence:.0%}"


def verdict_label(verdict):
    """Full verdict with the reason and confidence (inspection inset)."""
    if verdict is None:
        return "NOT INSPECTED", GREY
    if verdict.defective:
        return f"DEFECT: {verdict.reason} ({confidence_text(verdict)})", RED
    return f"OK ({confidence_text(verdict)})", GREEN


def heat_overlay(img, heat, alpha=0.55):
    """Blend a hot-coloured heatmap (values 0..1) over an image."""
    h = np.clip(heat, 0, 1)[..., None]
    color = np.concatenate([np.clip(3 * h, 0, 1), np.clip(3 * h - 1, 0, 1), np.clip(3 * h - 2, 0, 1)], axis=-1) * 255
    w = alpha * h
    return (img * (1 - w) + color * w).astype(np.uint8)


def draw_bottle_labels(frame, env, tracks, cam_name, active=None, label_font=None, tags=True):
    """Box + label over each tracked bottle. `active` = id of the track being picked.
    tags=False draws coloured boxes only (for small views where tags would overlap)."""
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]
    for t in tracks:
        box = track_box(env, t, cam_name, h, w)
        if box is None:
            continue
        x1, y1, x2, y2 = box
        text, color = short_label(t.verdict)
        if t.id == active and t.verdict is not None and t.verdict.defective:
            text = f"REJECT: {t.verdict.reason}"
        text = f"{t.label} {text}"
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3 if t.id == active else 2)
        if not tags:
            continue
        if label_font is None:
            draw.rectangle([x1, y1 - 13, x1 + len(text) * 6 + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - 12), text, fill=(255, 255, 255))
        else:
            tw, th = draw.textlength(text, font=label_font), label_font.size + 4
            lx = int(min(max(0, x1), w - tw - 6))
            draw.rectangle([lx, y1 - th, lx + tw + 6, y1], fill=color)
            draw.text((lx + 3, y1 - th + 1), text, fill=(255, 255, 255), font=label_font)
    return np.array(img)


def draw_inspection_inset(frame, station, size=88):
    """Top-left inset with the latest pair of inspection photos and the verdict."""
    if station.last is None:
        return frame
    label, front, back, verdict = station.last
    img = Image.fromarray(frame)
    thumbs = [Image.fromarray(v).resize((size, size)) for v in (front, back)]
    x0, y0 = 6, 32
    draw = ImageDraw.Draw(img)
    draw.rectangle([x0 - 2, y0 - 2, x0 + 2 * size + 6, y0 + size + 30], fill=(10, 10, 15))
    for i, t in enumerate(thumbs):
        img.paste(t, (x0 + i * (size + 4), y0))
    text, color = verdict_label(verdict)
    draw.text((x0, y0 + size + 2), f"INSPECTION  bottle {label}", fill=(200, 200, 200))
    draw.text((x0, y0 + size + 15), text, fill=color)
    return np.array(img)


def draw_reject_panel(frame, station, track_id, size=112):
    """Top-left panel while a bottle is being rejected: its inspection photos, the reason,
    the confidence and, for model decisions, a Grad-CAM heatmap of where the model saw it."""
    verdict = station.verdicts.get(track_id)
    if verdict is None or track_id not in station.photos:
        return frame
    photos = list(station.photos[track_id])
    if verdict.heatmaps is not None:
        photos = [heat_overlay(p, h) for p, h in zip(photos, verdict.heatmaps)]
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    x0, y0 = 6, 32
    w = 2 * size + 12
    lines = [f"REJECTING bottle #{track_id}", f"Reason: {verdict.reason}", f"Confidence: {confidence_text(verdict)}"]
    if verdict.heatmaps is not None:
        lines.append("Heat = where the model saw it")
    h = size + 10 + 14 * len(lines)
    draw.rectangle([x0 - 3, y0 - 3, x0 + w, y0 + h], fill=(25, 8, 8), outline=RED, width=2)
    for i, p in enumerate(photos):
        img.paste(Image.fromarray(p).resize((size, size)), (x0 + 2 + i * (size + 6), y0 + 2))
    for k, line in enumerate(lines):
        draw.text((x0 + 4, y0 + size + 6 + 14 * k), line,
                  fill=(255, 90, 80) if k < 2 else (220, 220, 220) if k == 2 else (170, 170, 170))
    return np.array(img)


def draw_hud(frame, title, status_text, step, counter_text, mode_label=None):
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    h, w = frame.shape[:2]
    draw.rectangle([0, 0, w, 26], fill=(10, 10, 15))
    draw.text((8, 4), title, fill=(0, 220, 180))
    draw.text((w - 230, 4), counter_text, fill=(255, 140, 60))
    if mode_label:
        color = {"VLA": (0, 255, 180), "ASSIST": (255, 200, 50)}.get(mode_label, (200, 200, 200))
        draw.text((w - 80, 4), f"[{mode_label}]", fill=color)
    draw.rectangle([0, h - 22, w, h], fill=(10, 10, 15))
    draw.text((8, h - 18), status_text, fill=(200, 200, 200))
    draw.text((w - 70, h - 18), f"step {step}", fill=(120, 120, 120))
    return np.array(img)


def compose_frame(env, obs, tracks, station, title, status, step, counter, mode_label=None, active=None,
                  cams=("sideview", "lineview"), images=None):
    """Side-by-side video frame, one panel per camera; the first panel carries the reject panel
    (or the latest inspection). `images` can supply upright camera images not in `obs`."""
    panels = []
    for k, cam in enumerate(cams):
        img = images[cam].copy() if images and cam in images else obs[f"{cam}_image"][::-1].copy()
        img = draw_bottle_labels(img, env, tracks, cam, active=active)
        if k == 0:
            # while a bottle is being rejected, show why; otherwise the latest inspection
            if active is not None and active in station.photos:
                img = draw_reject_panel(img, station, active)
            else:
                img = draw_inspection_inset(img, station)
        panels.append(img)
    return draw_hud(np.concatenate(panels, axis=1), title, status, step, counter, mode_label)


LIVE_CAMERA = "agentview"   # the robot's front view, as in robosuite's own viewer

# Live window layout (pixels): header, big front view on the left, and a side column with
# the inspection panel, the bottle detector's view and the line overview; footer with the
# status line.
LIVE_W, LIVE_H = 1600, 900
HEADER, FOOTER, GAP = 44, 40, 4
SIDE_W = 496
MAIN_W = LIVE_W - SIDE_W - GAP
BODY_H = LIVE_H - HEADER - FOOTER
TILE_H = (BODY_H - 2 * GAP) // 3
BG, PANEL_BG, MUTED = (14, 15, 20), (24, 26, 34), (150, 155, 165)
CAM_CAPTIONS = {"agentview": "ROBOT FRONT VIEW", "sideview": "SIDE VIEW", "lineview": "LINE OVERVIEW"}


def _caption(img, text):
    """Small caption chip in the top-left corner of a camera tile."""
    draw = ImageDraw.Draw(img, "RGBA")
    f = font(13, bold=True)
    draw.rectangle([0, 0, draw.textlength(text, font=f) + 14, 22], fill=(0, 0, 0, 150))
    draw.text((7, 3), text, fill=(230, 230, 230), font=f)


def _camera_tile(env, tracks, cam, w, h, active, tags=True):
    img = env.sim.render(width=w, height=h, camera_name=cam)[::-1].copy()
    img = Image.fromarray(draw_bottle_labels(img, env, tracks, cam, active=active,
                                             label_font=font(15, bold=True), tags=tags))
    _caption(img, CAM_CAPTIONS.get(cam, cam.upper()))
    return img


def _detector_tile(env, vision, w, h):
    """What the bottle detector sees: the overhead camera image with its detections and the
    tracker's IDs. Downstream (towards the robot) is on the left."""
    if vision is None or vision.last_rgb is None:
        tile = Image.new("RGB", (w, h), PANEL_BG)
        _caption(tile, "OVERHEAD CAMERA - BOTTLE DETECTOR")
        return tile
    src_h, src_w = vision.last_rgb.shape[:2]
    scale = min(w / src_w, h / src_h)
    img = Image.fromarray(vision.last_rgb).resize((int(src_w * scale), int(src_h * scale)))
    tile = Image.new("RGB", (w, h), (0, 0, 0))
    ox, oy = (w - img.width) // 2, (h - img.height) // 2
    tile.paste(img, (ox, oy))
    draw = ImageDraw.Draw(tile)
    f = font(12, bold=True)
    for d in vision.last_detections:
        x1, y1, x2, y2 = d.box * scale + np.array([ox, oy, ox, oy])
        draw.rectangle([x1, y1, x2, y2], outline=CYAN if d.upright else (255, 0, 255), width=2)
    for t in vision.tracks:
        pts = project(env.sim, np.array([[t.xy[0], t.xy[1], t.top_z]]), vision.camera.name, src_h, src_w)
        if pts is None:
            continue
        px, py = pts[0] * scale + np.array([ox, oy])
        text, color = short_label(t.verdict)
        text = f"{t.label} {text}"
        tw = draw.textlength(text, font=f)
        draw.rectangle([px - tw / 2 - 3, py - 30, px + tw / 2 + 3, py - 15], fill=color)
        draw.text((px - tw / 2, py - 29), text, fill=(255, 255, 255), font=f)
    _caption(tile, "OVERHEAD RGB-D CAMERA - BOTTLE DETECTOR")
    return tile


def _inspection_tile(station, active, w, h):
    """Side panel: the bottle being rejected (photos + heatmap, reason, confidence), or else the
    latest inspection. Drawn in its own tile so it never covers a camera view."""
    tile = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(tile)
    rejecting = active is not None and active in station.photos and station.verdicts.get(active) is not None
    if rejecting:
        verdict = station.verdicts[active]
        photos = list(station.photos[active])
        if verdict.heatmaps is not None:
            photos = [heat_overlay(p, hm) for p, hm in zip(photos, verdict.heatmaps)]
        heading, head_bg = f"REJECTING  bottle #{active}", RED
    elif station.last is not None:
        label, front, back, verdict = station.last
        photos = [front, back]
        heading, head_bg = f"LAST INSPECTED  bottle {label}", (45, 50, 62)
    else:
        draw.rectangle([0, 0, w, 30], fill=(45, 50, 62))
        draw.text((10, 6), "INSPECTION STATION", fill=(255, 255, 255), font=font(15, bold=True))
        draw.text((10, 50), "Waiting for the first bottle...", fill=MUTED, font=font(15))
        return tile

    draw.rectangle([0, 0, w, 30], fill=head_bg)
    draw.text((10, 6), heading, fill=(255, 255, 255), font=font(15, bold=True))

    size = min(150, h - 30 - 80)
    for i, p in enumerate(photos):
        x = 10 + i * (size + 8)
        tile.paste(Image.fromarray(p).resize((size, size)), (x, 38))
        draw.text((x + 4, 38 + size - 17), ("front", "back")[i], fill=(255, 255, 255), font=font(12))

    # verdict badge right of the photos
    bx0, by0 = 10 + 2 * (size + 8), 38
    bad = verdict.defective
    draw.rectangle([bx0, by0, w - 10, by0 + size], outline=RED if bad else GREEN, width=3)
    cx = (bx0 + w - 10) / 2
    conf = confidence_text(verdict)
    for text, f, y, col in (("REJECT" if bad else "OK", font(26, bold=True), by0 + 30, RED if bad else GREEN),
                            (conf, font(22 if len(conf) < 6 else 15, bold=True), by0 + 72, (235, 235, 235)),
                            ("confidence" if len(conf) < 6 else "decision", font(12), by0 + 102, MUTED)):
        draw.text((cx - draw.textlength(text, font=f) / 2, y), text, fill=col, font=f)

    y = 38 + size + 10
    if bad:
        draw.text((10, y), f"Reason: {verdict.reason}", fill=(255, 110, 100), font=font(18, bold=True))
        if rejecting and verdict.heatmaps is not None:
            draw.text((10, y + 26), "Heatmap = where the model saw the defect", fill=MUTED, font=font(13))
    else:
        draw.text((10, y), "No defect found - bottle goes to the carton", fill=GREEN, font=font(16, bold=True))
    return tile


def live_frame(env, vision, station, title, status, step, counter, mode_label=None, active=None,
               overview="lineview"):
    """Annotated dashboard frame for the live window (LIVE_W x LIVE_H)."""
    tracks = vision.tracks if vision is not None else []
    canvas = Image.new("RGB", (LIVE_W, LIVE_H), BG)
    canvas.paste(_camera_tile(env, tracks, LIVE_CAMERA, MAIN_W, BODY_H, active), (0, HEADER))
    x = MAIN_W + GAP
    canvas.paste(_inspection_tile(station, active, SIDE_W, TILE_H), (x, HEADER))
    canvas.paste(_detector_tile(env, vision, SIDE_W, TILE_H), (x, HEADER + TILE_H + GAP))
    canvas.paste(_camera_tile(env, tracks, overview, SIDE_W, TILE_H, active, tags=False),
                 (x, HEADER + 2 * (TILE_H + GAP)))

    draw = ImageDraw.Draw(canvas)
    draw.text((14, 11), title, fill=(0, 220, 180), font=font(20, bold=True))
    right = LIVE_W - 14
    if mode_label:
        color = {"VLA": (0, 255, 180), "ASSIST": (255, 200, 50)}.get(mode_label, (200, 200, 200))
        text = f"[{mode_label}]"
        right -= draw.textlength(text, font=font(18, bold=True))
        draw.text((right, 12), text, fill=color, font=font(18, bold=True))
        right -= 24
    draw.text((right - draw.textlength(counter, font=font(18, bold=True)), 12), counter,
              fill=(255, 150, 70), font=font(18, bold=True))

    fy = LIVE_H - FOOTER
    bad = status.upper().startswith("REJECT")
    draw.rectangle([0, fy, LIVE_W, LIVE_H], fill=(70, 18, 18) if bad else BG)
    draw.text((14, fy + 9), status, fill=(255, 255, 255) if bad else (210, 210, 210), font=font(18, bold=True))
    step_text = f"step {step}"
    draw.text((LIVE_W - 14 - draw.textlength(step_text, font=font(15)), fy + 11), step_text,
              fill=(120, 120, 120), font=font(15))
    return np.array(canvas)


class LiveWindow:
    """OpenCV window showing annotated frames while the simulation runs. robosuite's own
    viewer only shows raw camera images, so it can't display the rejection reasons."""

    def __init__(self, name="Inspection line (live)", size=(LIVE_W, LIVE_H)):
        import cv2
        self.cv2, self.name = cv2, name
        # GUI_NORMAL hides the Qt toolbar/status bar; KEEPRATIO keeps the frame undistorted
        cv2.namedWindow(name, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow(name, *size)

    def show(self, frame):
        self.cv2.imshow(self.name, np.ascontiguousarray(frame[..., ::-1]))
        self.cv2.waitKey(1)

    def close(self):
        self.cv2.destroyWindow(self.name)
