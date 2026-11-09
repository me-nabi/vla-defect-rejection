"""Randomized appearance and defects for `inspection_bottle` objects.

Each inspection bottle is one of the designs in bottle_shapes.py and carries the geoms
written by tools/make_inspection_bottle.py. At reset the environment picks which bottles are
defective. This module then edits their geoms in the compiled model: colour, size and pose of
visual-only geoms, which never affects physics.

Defect types, each with randomized size, position and shape:
    crack        thin pale fracture line on the glass                   (glass bottles)
    chip         small bright fractured flake at the heel or upper body (glass bottles)
    dented       the plastic body is pushed in (three dent shapes)      (PET bottles)
    missing_cap  no cap
    tilted_cap   cap sitting askew
    low_fill     liquid level clearly below the normal fill line
    particle     dark foreign particle floating in the liquid
Ranges run from obvious to subtle (a 2-segment hairline crack, a 1 mm particle, a cap
tilted by 5 degrees, a fill level just under spec).

Any bottle, good or defective, also gets natural variation and harmless look-alikes,
so a detector has to learn the defects, not just "anything unusual": product colour
(orange, apple, berry, lime, cola, milk), glass tint (clear, green, amber), cap and label
colour, fill level within spec, label shifted or rotated, air bubbles in the drink (vs.
particles), condensation droplets on the glass (vs. chips).
"""
import numpy as np
import robosuite.utils.transform_utils as T

from libero.libero.envs.bottle_shapes import ALL_DEFECTS

DEFECT_TYPES = ALL_DEFECTS

LIQUID_Z0 = 0.004
N_CRACK_SEGMENTS, N_CHIP_FLAKES = 8, 3
N_BUBBLES, N_DROPS = 6, 12
P_BUBBLES, P_DROPS, P_LABEL_SHIFT = 0.5, 0.4, 0.5   # chance of each harmless feature per bottle

LIQUIDS = {           # rgba; alpha = how cloudy the drink is
    "orange": (0.96, 0.55, 0.10, 0.50),
    "apple": (0.92, 0.76, 0.28, 0.45),
    "berry": (0.62, 0.07, 0.16, 0.62),
    "lime": (0.52, 0.78, 0.18, 0.45),
    "cola": (0.28, 0.13, 0.05, 0.82),
    "milk": (0.95, 0.93, 0.88, 0.88),
}
GLASS_TINTS = {"clear": (0.82, 0.93, 0.90, 0.30), "green": (0.35, 0.62, 0.38, 0.40),
               "amber": (0.62, 0.40, 0.14, 0.42)}
PET_TINTS = {"clear": (0.88, 0.93, 0.97, 0.18), "blue": (0.72, 0.85, 0.97, 0.24)}
LIGHT_LIQUIDS = ("orange", "apple", "lime", "milk")      # the only ones put in tinted glass
CAPS = [(0.12, 0.32, 0.72), (0.75, 0.10, 0.10), (0.93, 0.93, 0.92), (0.08, 0.08, 0.09),
        (0.10, 0.50, 0.22), (0.80, 0.65, 0.25), (0.95, 0.48, 0.08)]
LABEL_TINTS = [(1, 1, 1), (1, 0.95, 0.84), (1, 0.84, 0.87), (0.84, 1, 0.90), (0.84, 0.91, 1), (1, 0.97, 0.72)]
FLAW_RGB = np.array([0.95, 0.97, 1.0])
HIDDEN = np.array([0.0, 0.0, 0.0, 0.0])


def _wxyz(mat):
    return T.convert_quat(T.mat2quat(mat), to="wxyz")


class BottleAppearance:
    """Handles to one bottle's geoms in a compiled model, and setters for its look."""

    def __init__(self, sim, name, shape):
        m = sim.model
        gid = lambda s: m.geom_name2id(f"{name}_{s}")
        self.name, self.shape = name, shape
        self.glass, self.cap, self.particle, self.label = gid("glass"), gid("cap"), gid("particle"), gid("label")
        self.liquid_lo, self.liquid_hi = gid("liquid_lo"), gid("liquid_hi")
        self.cracks = [gid(f"crack{i}") for i in range(N_CRACK_SEGMENTS)]
        self.chips = [gid(f"chip{i}") for i in range(N_CHIP_FLAKES)]
        self.bubbles = [gid(f"bubble{i}") for i in range(N_BUBBLES)]
        self.drops = [gid(f"drop{i}") for i in range(N_DROPS)]
        self.dents = [(gid(f"dent{k}"), gid(f"dent_liquid{k}")) for k in range(len(shape.dents))]
        self.cap_pos = m.geom_pos[self.cap].copy()
        self.cap_quat = m.geom_quat[self.cap].copy()
        self.label_pos = m.geom_pos[self.label].copy()     # compiled mesh offsets
        self.label_quat = m.geom_quat[self.label].copy()
        # compiled mesh frames (MuJoCo moves each mesh to its inertial frame)
        self.dent_frames = [[(m.geom_pos[g].copy(), m.geom_quat[g].copy()) for g in pair] for pair in self.dents]
        # MuJoCo 3 skips the frame computation for geoms compiled at their body's frame
        # ("sameframe"), so later geom_pos/geom_quat edits would be ignored. These geoms move.
        if hasattr(m, "geom_sameframe"):
            moving = [self.liquid_hi, self.cap, self.particle, self.label] + self.cracks + self.chips \
                + self.bubbles + self.drops + [g for pair in self.dents for g in pair]
            for g in moving:
                m.geom_sameframe[g] = 0

    # -- building blocks -------------------------------------------------------------
    def _set(self, m, g, rgba=None, pos=None, quat=None, size=None):
        if rgba is not None:
            m.geom_rgba[g] = rgba
        if pos is not None:
            m.geom_pos[g] = pos
        if quat is not None:
            m.geom_quat[g] = quat
        if size is not None:
            m.geom_size[g] = size
            m.geom_rbound[g] = float(np.linalg.norm(size[:3]))

    def _set_fill(self, m, level):
        z0 = self.shape.liquid_split
        half = max(level - z0, 0.0005) / 2
        self._set(m, self.liquid_hi, pos=[0, 0, z0 + half], size=[self.shape.liquid_r_hi, half, 0])
        self.fill_top = level

    def _surface_frame(self, phi, z, offset=0.0):
        """Point on the outside surface plus the tangent (e_phi), vertical and outward-normal axes."""
        r = float(self.shape.outer_radius(z)) + offset
        n = np.array([np.cos(phi), np.sin(phi), 0.0])
        e_phi = np.array([-np.sin(phi), np.cos(phi), 0.0])
        return np.array([r * np.cos(phi), r * np.sin(phi), z]), e_phi, np.array([0.0, 0.0, 1.0]), n

    def _liquid_zones(self, fill_top):
        return [(lo, (fill_top - 0.005) if hi is None else hi) for lo, hi in self.shape.particle_zones
                if ((fill_top - 0.005) if hi is None else hi) > lo + 0.002]

    def _reset_look(self, m, rng):
        """A good bottle of this design, with random product colours and small variation."""
        s = self.shape
        if s.material == "pet":
            glass = np.array(PET_TINTS[rng.choice(list(PET_TINTS))], float)
            liquid_name = rng.choice(list(LIQUIDS))
        else:
            tint = rng.choice(list(GLASS_TINTS), p=[0.6, 0.2, 0.2])
            glass = np.array(GLASS_TINTS[tint], float)
            liquid_name = rng.choice(list(LIQUIDS) if tint == "clear" else list(LIGHT_LIQUIDS))
        glass[:3] = np.clip(glass[:3] + rng.uniform(-0.03, 0.03, 3), 0, 1)
        glass[3] = np.clip(glass[3] + rng.uniform(-0.04, 0.04), 0.12, 0.6)
        liquid = np.array(LIQUIDS[liquid_name], float)
        liquid[:3] = np.clip(liquid[:3] + rng.uniform(-0.04, 0.04, 3), 0, 1)
        self.colors = dict(glass=glass, liquid=liquid)
        for g in (self.glass,):
            self._set(m, g, rgba=glass)
        for g in (self.liquid_lo, self.liquid_hi):
            self._set(m, g, rgba=liquid)
        for body, liq in self.dents:
            self._set(m, body, rgba=HIDDEN)
            self._set(m, liq, rgba=HIDDEN)
        self._set_fill(m, rng.uniform(*s.normal_fill))
        cap = np.r_[np.clip(np.array(CAPS[rng.integers(len(CAPS))]) + rng.uniform(-0.03, 0.03, 3), 0, 1), 1.0]
        self._set(m, self.cap, rgba=cap, pos=self.cap_pos, quat=self.cap_quat)
        for g in [self.particle] + self.cracks + self.chips + self.bubbles + self.drops:
            self._set(m, g, rgba=HIDDEN)
        self._label(m, rng, shift=rng.random() < P_LABEL_SHIFT)
        if rng.random() < P_BUBBLES:
            self._bubbles(m, rng)
        if rng.random() < P_DROPS:
            self._drops(m, rng)

    # -- harmless variation ------------------------------------------------------------
    def _label(self, m, rng, shift):
        """Label tinted, rotated about the bottle axis and, when `shift`, a little off-height."""
        tint = np.array(LABEL_TINTS[rng.integers(len(LABEL_TINTS))], float)
        yaw = rng.uniform(0, 2 * np.pi)
        rz = T.quat2mat(T.axisangle2quat(np.array([0.0, 0.0, yaw])))
        dz = rng.uniform(-0.0015, 0.0015) if shift else 0.0
        q = T.quat_multiply(T.axisangle2quat(np.array([0.0, 0.0, yaw])), T.convert_quat(self.label_quat, to="xyzw"))
        self._set(m, self.label, rgba=np.r_[tint, 1.0], pos=rz @ self.label_pos + np.array([0, 0, dz]),
                  quat=T.convert_quat(q, to="wxyz"))

    def _bubbles(self, m, rng):
        zones = self._liquid_zones(self.fill_top)
        n = int(rng.integers(1, N_BUBBLES + 1))
        for i, g in enumerate(self.bubbles):
            if i >= n or not zones:
                continue
            if rng.random() < 0.6:                     # most bubbles collect under the surface
                z = rng.uniform(self.fill_top - 0.004, self.fill_top - 0.0005)
            else:
                lo, hi = zones[rng.integers(len(zones))]
                z = rng.uniform(lo, hi)
            r_max = float(self.shape.inner_radius(z)) - 0.0015
            r, phi = rng.uniform(0.0, r_max), rng.uniform(0, 2 * np.pi)
            self._set(m, g, rgba=[0.97, 0.97, 0.95, rng.uniform(0.45, 0.8)],
                      pos=[r * np.cos(phi), r * np.sin(phi), z], size=[rng.uniform(0.0004, 0.0012), 0, 0])

    def _drops(self, m, rng):
        n = int(rng.integers(3, N_DROPS + 1))
        for i, g in enumerate(self.drops):
            if i >= n:
                continue
            p, e_phi, e_z, nrm = self._surface_frame(rng.uniform(0, 2 * np.pi),
                                                     rng.uniform(0.003, self.shape.z_shoulder - 0.002), 0.0002)
            rot = np.stack([e_phi, e_z, nrm], axis=1)
            d = rng.uniform(0.0004, 0.001)
            self._set(m, g, rgba=[0.9, 0.95, 1.0, rng.uniform(0.3, 0.6)], pos=p, quat=_wxyz(rot),
                      size=[d, d * rng.uniform(0.8, 1.3), 0.00025])

    # -- defects ---------------------------------------------------------------------
    def _crack(self, m, rng):
        bands = self.shape.crack_bands
        lo, hi = bands[0] if rng.random() < 0.8 else bands[-1]
        n_seg = int(rng.integers(2, N_CRACK_SEGMENTS + 1))
        s, z = 0.0, rng.uniform(lo, hi)            # arc length along the circumference, height
        phi0 = rng.uniform(0, 2 * np.pi)
        angle = rng.uniform(-1.0, 1.0) + (np.pi if rng.random() < 0.5 else 0.0)
        width = rng.uniform(0.0002, 0.0005)
        alpha = rng.uniform(0.35, 0.85)
        r_ref = self.shape.r_body
        for i, g in enumerate(self.cracks):
            if i >= n_seg:
                self._set(m, g, rgba=HIDDEN)
                continue
            seg = rng.uniform(0.003, 0.006)
            angle += rng.normal(0, 0.5)
            ds, dz = seg * np.cos(angle), seg * np.sin(angle)
            if not lo <= z + dz <= hi:             # bounce off the band edges
                angle, dz = -angle, -dz
            sm, zm = s + ds / 2, z + dz / 2
            p, e_phi, e_z, n = self._surface_frame(phi0 + sm / r_ref, zm, 0.0001)
            t = np.cos(angle) * e_phi + np.sin(angle) * e_z
            rot = np.stack([t, np.cross(n, t), n], axis=1)
            self._set(m, g, rgba=np.r_[FLAW_RGB, alpha], pos=p, quat=_wxyz(rot), size=[seg / 2 + 0.0003, width, 0.0002])
            s, z = s + ds, z + dz

    def _chip(self, m, rng):
        heel, upper = self.shape.chip_zones
        lo, hi = heel if rng.random() < 0.5 else upper
        z = rng.uniform(lo, hi)
        phi = rng.uniform(0, 2 * np.pi)
        for i, g in enumerate(self.chips):
            if i > 0 and rng.random() < 0.3:
                self._set(m, g, rgba=HIDDEN)
                continue
            dphi, dz = rng.uniform(-0.08, 0.08), rng.uniform(-0.0012, 0.0012)
            p, e_phi, e_z, n = self._surface_frame(phi + dphi, np.clip(z + dz, 0.001, self.shape.z_shoulder - 0.002), -0.0002)
            spin = rng.uniform(0, np.pi)
            t = np.cos(spin) * e_phi + np.sin(spin) * e_z
            rot = np.stack([t, np.cross(n, t), n], axis=1)
            size = [rng.uniform(0.001, 0.003), rng.uniform(0.0009, 0.0025), rng.uniform(0.0005, 0.0009)]
            self._set(m, g, rgba=np.r_[FLAW_RGB, rng.uniform(0.5, 0.95)], pos=p, quat=_wxyz(rot), size=size)

    def _tilted_cap(self, m, rng):
        tilt = np.radians(rng.uniform(5, 22))
        axis_angle = rng.uniform(0, 2 * np.pi)
        axis = np.array([np.cos(axis_angle), np.sin(axis_angle), 0.0])
        rot = T.quat2mat(T.axisangle2quat(axis * tilt))
        side = np.cross(axis, [0, 0, 1.0])      # the cap's raised edge leans this way
        pos = self.cap_pos + side * rng.uniform(0.001, 0.002) + np.array([0, 0, rng.uniform(0.0015, 0.003)])
        self._set(m, self.cap, pos=pos, quat=_wxyz(rot))

    def _particle(self, m, rng):
        zones = self._liquid_zones(self.fill_top)
        lo, hi = zones[rng.integers(len(zones))]
        z = rng.uniform(lo, hi)
        r_max = float(self.shape.inner_radius(z)) - 0.0025
        r, phi = rng.uniform(0.004, max(r_max, 0.0045)), rng.uniform(0, 2 * np.pi)
        shade = rng.uniform(0.04, 0.35)
        self._set(m, self.particle, rgba=[shade, shade * 0.85, shade * 0.7, 1.0],
                  pos=[r * np.cos(phi), r * np.sin(phi), z], size=[rng.uniform(0.0008, 0.0025), 0, 0])

    def _dented(self, m, rng):
        """Swap the body and liquid for one of the pre-made dented ones, turned to a random
        side. The dented liquid is filled to the design's normal level."""
        k = int(rng.integers(len(self.dents)))
        q_yaw = T.axisangle2quat(np.array([0.0, 0.0, rng.uniform(0, 2 * np.pi)]))     # xyzw
        rz = T.quat2mat(q_yaw)
        for g in (self.glass, self.liquid_lo, self.liquid_hi):
            self._set(m, g, rgba=HIDDEN)
        for g, (pos, quat), color in zip(self.dents[k], self.dent_frames[k], (self.colors["glass"], self.colors["liquid"])):
            q = T.quat_multiply(q_yaw, T.convert_quat(quat, to="xyzw"))
            self._set(m, g, rgba=color, pos=rz @ pos, quat=T.convert_quat(q, to="wxyz"))
        self.fill_top = self.shape.dent_fill
        for g in self.bubbles:                     # bubbles sat under the old surface
            self._set(m, g, rgba=HIDDEN)

    # -- public ----------------------------------------------------------------------
    def apply(self, sim, defect=None, rng=None):
        """Give this bottle a fresh random look, with `defect` (one of shape.defects) or none.

        rng: a numpy Generator (default: a fresh one).
        """
        rng = np.random.default_rng() if rng is None else rng
        m = sim.model
        if defect is not None and defect not in self.shape.defects:
            raise ValueError(f"{defect!r} is not possible on a {self.shape.name} bottle; "
                             f"expected one of {self.shape.defects}")
        self._reset_look(m, rng)
        if defect is None:
            return
        if defect == "crack":
            self._crack(m, rng)
        elif defect == "chip":
            self._chip(m, rng)
        elif defect == "dented":
            self._dented(m, rng)
        elif defect == "missing_cap":
            self._set(m, self.cap, rgba=HIDDEN)
        elif defect == "tilted_cap":
            self._tilted_cap(m, rng)
        elif defect == "low_fill":
            self._set_fill(m, rng.uniform(*self.shape.low_fill))
        elif defect == "particle":
            self._particle(m, rng)
