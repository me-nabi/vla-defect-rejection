"""Visual-only warehouse dressing around the inspection line.

Everything here has contype = conaffinity = 0, so physics is unchanged.
Placement rule: stay out of the two inspection cameras' field of view. They look across the
belt at x = 0.20 (fovy 25 deg) and see the walls behind; anything new in their images would
change what the trained classifier sees. Props therefore keep |x - 0.2| > 0.8 near the side
walls, and the inspection gantry posts stand outside the cameras' cone.
Also mind robosuite's render budget (MjvScene maxgeom = 1000; the scene uses ~360).
"""
import os

import mujoco
import numpy as np
from robosuite.utils.mjcf_utils import array_to_string, new_element

TEX_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        "..", "..", "assets", "conveyor_line", "textures"))


def yaw_quat(yaw):
    return [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]


def facing_quat(facing):
    """Quaternion (wxyz) turning a plate's local z towards `facing` ("+x", "-x", "+y", "-y"),
    with local x to the viewer's right and local y up, so a texture reads correctly."""
    normal = {"+x": (1, 0, 0), "-x": (-1, 0, 0), "+y": (0, 1, 0), "-y": (0, -1, 0)}[facing]
    z = np.array(normal, float)
    y = np.array([0.0, 0.0, 1.0])
    x = np.cross(y, z)                      # viewer's right, seen from the +normal side
    mat = np.stack([x, y, z], axis=1)
    quat = np.zeros(4)
    mujoco.mju_mat2Quat(quat, mat.flatten())
    return list(quat)


class PropBuilder:
    """Appends named visual geoms, textures and materials to an arena (or a body in it)."""

    def __init__(self, arena, prefix="wh"):
        self.arena, self.prefix, self.n = arena, prefix, 0

    def material(self, name, texture=None, texrepeat=None, texuniform=False, **attrs):
        mat = dict(name=f"{self.prefix}_{name}", **{k: str(v) for k, v in attrs.items()})
        if texture:
            tex_name = f"{self.prefix}_{name}_tex"
            self.arena.asset.append(new_element(tag="texture", name=tex_name, type="2d",
                                                file=os.path.join(TEX_DIR, texture)))
            mat.update(texture=tex_name, texuniform=str(texuniform).lower())
            if texrepeat:
                mat["texrepeat"] = array_to_string(texrepeat)
        self.arena.asset.append(new_element(tag="material", **mat))
        return mat["name"]

    def geom(self, type, pos, size, material=None, rgba=None, quat=None, parent=None):
        self.n += 1
        attrs = dict(name=f"{self.prefix}_g{self.n}", type=type, pos=array_to_string(pos),
                     size=array_to_string(size), group="1", contype="0", conaffinity="0")
        if quat is not None:
            attrs["quat"] = array_to_string(quat)
        if material:
            attrs["material"] = material
        else:
            attrs["rgba"] = array_to_string(rgba if rgba is not None else [0.5, 0.5, 0.5, 1])
        (parent if parent is not None else self.arena.worldbody).append(new_element(tag="geom", **attrs))

    def box(self, pos, half, yaw=0.0, **kw):
        self.geom("box", pos, half, quat=yaw_quat(yaw) if yaw else kw.pop("quat", None), **kw)

    def cylinder(self, pos, radius, half_h, **kw):
        self.geom("cylinder", pos, [radius, half_h], **kw)

    def sign(self, pos, half_w, half_h, facing, material, thick=0.003, parent=None):
        """Flat textured plate. MuJoCo projects a 2D texture through a box's local XY plane, so
        the plate is thin along local z, and local z is turned towards the viewer."""
        self.geom("box", pos, [half_w, half_h, thick], material=material, quat=facing_quat(facing), parent=parent)


class WarehouseProps:
    """Racking with cartons, floor pallets, a pallet truck, a robot-cell fence with a caution
    sign, an inspection gantry with a sign, a control cabinet with a stack light, drums and
    floor markings."""

    def __init__(self, arena, seed=7):
        self.b = PropBuilder(arena)
        self.rng = np.random.default_rng(seed)
        b = self.b
        self.m = dict(
            carton=b.material("carton", texture="carton.png", specular=0.05, shininess=0.05),
            wood=b.material("wood", texture="pallet_wood.png", specular=0.02, shininess=0.02),
            hazard=b.material("hazard", texture="hazard.png", specular=0.1, shininess=0.1),
            upright=b.material("upright", rgba="0.12 0.26 0.56 1", specular=0.4, shininess=0.4),
            beam=b.material("beam", rgba="0.92 0.42 0.06 1", specular=0.4, shininess=0.4),
            yellow=b.material("yellow", rgba="0.93 0.72 0.05 1", specular=0.3, shininess=0.3),
            steel=b.material("steel", rgba="0.55 0.56 0.58 1", specular=0.6, shininess=0.6),
            dark=b.material("dark", rgba="0.10 0.10 0.11 1", specular=0.2, shininess=0.2),
            mesh=b.material("mesh", rgba="0.16 0.17 0.18 0.55", specular=0.1, shininess=0.1),
            cabinet=b.material("cabinet", rgba="0.78 0.79 0.77 1", specular=0.3, shininess=0.3),
            screen=b.material("screen", rgba="0.15 0.45 0.75 1", emission=0.8),
            drum=b.material("drum", rgba="0.10 0.32 0.65 1", specular=0.5, shininess=0.5),
            red=b.material("red", rgba="0.80 0.08 0.06 1", specular=0.5, shininess=0.5),
            lamp_r=b.material("lamp_r", rgba="0.9 0.1 0.1 1", emission=0.3),
            lamp_y=b.material("lamp_y", rgba="0.95 0.7 0.1 1", emission=0.3),
            lamp_g=b.material("lamp_g", rgba="0.1 0.95 0.25 1", emission=1.0),
            light_strip=b.material("light_strip", rgba="0.95 0.97 1.0 1", emission=1.0),
            sign_inspection=b.material("sign_inspection", texture="sign_inspection.png", emission=0.25),
            sign_caution=b.material("sign_caution", texture="sign_caution.png", emission=0.15),
            sign_packing=b.material("sign_packing", texture="sign_packing.png", emission=0.2),
        )

    def build(self, belt_exit=None):
        self.racking(x_front=-1.25, x_back=-1.93, ys=(-2.4, -1.2, 0.0, 1.2, 2.4), levels=(0.12, 0.85, 1.55))
        for (x, y, yaw) in ((1.75, 2.2, 0.1), (-0.95, 2.4, 0.0), (1.65, -2.35, -0.05), (2.3, 1.2, 1.57)):
            self.pallet_with_cartons(x, y, 0.0, yaw=yaw, layers=int(self.rng.integers(2, 4)))
        self.pallet_truck(-0.9, 1.7, yaw=np.pi / 2)
        self.fence(x=-1.12, y0=-0.55, y1=0.85)
        self.inspection_gantry(x_post=0.58, x_tip=-0.08, y=0.42, top=1.64)
        self.cabinet(1.6, 0.62)
        self.drums([(1.25, -2.6), (1.5, -2.62), (1.37, -2.38)])
        self.floor_markings()
        if belt_exit is not None:
            crate_y = belt_exit.exit_y_end - belt_exit.crate_half - 0.012
            self.packing_sign(belt_exit.exit_x - belt_exit.crate_half - 0.3, crate_y)
        return self.b.n

    # ---- pieces ------------------------------------------------------------------------
    def pallet(self, x, y, z, sx=0.6, sy=0.5, yaw=0.0):
        """Wooden pallet, footprint 2*sx by 2*sy, standing on z; returns its top height."""
        b, c, s = self.b, np.cos(yaw), np.sin(yaw)
        rot = lambda dx, dy: (x + c * dx - s * dy, y + s * dx + c * dy)
        h_block, h_board = 0.05, 0.011
        for dx in (-sx + 0.05, 0.0, sx - 0.05):          # three stringers of blocks
            px, py = rot(dx, 0.0)
            b.box([px, py, z + h_block / 2], [0.045, sy, h_block / 2], yaw, material=self.m["wood"])
        for dy in np.linspace(-sy + 0.04, sy - 0.04, 5):  # top deck boards
            px, py = rot(0.0, dy)
            b.box([px, py, z + h_block + h_board], [sx, 0.045, h_board], yaw, material=self.m["wood"])
        return z + h_block + 2 * h_board

    def cartons(self, x, y, z, sx, sy, yaw, layers):
        """Stack of cartons on a footprint; the top layer is sometimes incomplete."""
        b, c, s = self.b, np.cos(yaw), np.sin(yaw)
        nx, ny = max(1, int(round(sx / 0.17))), max(1, int(round(sy / 0.15)))
        hx, hy = sx / nx - 0.006, sy / ny - 0.006
        hz = float(self.rng.uniform(0.09, 0.13))
        for k in range(layers):
            for i in range(nx):
                for j in range(ny):
                    if k == layers - 1 and self.rng.random() < 0.3:
                        continue
                    dx = -sx + (2 * i + 1) * sx / nx + self.rng.normal(0, 0.004)
                    dy = -sy + (2 * j + 1) * sy / ny + self.rng.normal(0, 0.004)
                    b.box([x + c * dx - s * dy, y + s * dx + c * dy, z + hz + 2 * hz * k], [hx, hy, hz],
                          yaw + self.rng.normal(0, 0.015), material=self.m["carton"])

    def pallet_with_cartons(self, x, y, z, yaw=0.0, layers=2, sx=0.6, sy=0.5):
        top = self.pallet(x, y, z, sx, sy, yaw)
        self.cartons(x, y, top, sx - 0.03, sy - 0.03, yaw, layers)

    def racking(self, x_front, x_back, ys, levels, height=2.1):
        b, m = self.b, self.m
        depth_half = (x_front - x_back) / 2
        xc = (x_front + x_back) / 2
        for y in ys:
            for x in (x_front, x_back):
                b.box([x, y, height / 2], [0.03, 0.035, height / 2], material=m["upright"])
            for z in (0.45, 1.2, 1.9):
                b.box([xc, y, z], [depth_half, 0.012, 0.012], material=m["upright"])
        for y0, y1 in zip(ys[:-1], ys[1:]):
            ym, half = (y0 + y1) / 2, (y1 - y0) / 2 - 0.035
            for z in levels:
                for x in (x_front, x_back):
                    b.box([x, ym, z], [0.025, half, 0.05], material=m["beam"])
                if z < levels[-1] or self.rng.random() < 0.7:
                    top = self.pallet(xc, ym, z + 0.05, sx=depth_half - 0.04, sy=min(half - 0.06, 0.5))
                    layers = 2 if z < levels[-1] else 1
                    self.cartons(xc, ym, top, depth_half - 0.07, min(half - 0.09, 0.47), 0.0, layers)

    def pallet_truck(self, x, y, yaw=0.0):
        """Hand pallet truck: two forks, a body with the hydraulic unit, a handle."""
        b, m, c, s = self.b, self.m, np.cos(yaw), np.sin(yaw)
        rot = lambda dx, dy: (x + c * dx - s * dy, y + s * dx + c * dy)
        for dy in (-0.19, 0.19):
            px, py = rot(0.0, dy)
            b.box([px, py, 0.045], [0.55, 0.08, 0.03], yaw, material=m["dark"])
        px, py = rot(-0.6, 0.0)
        b.box([px, py, 0.12], [0.08, 0.28, 0.1], yaw, material=m["yellow"])
        b.cylinder([px, py, 0.32], 0.05, 0.12, material=m["dark"])
        hx, hy = rot(-0.74, 0.0)
        tilt = -np.radians(20)     # handle leans back, about the truck's lateral (local y) axis
        q = _quat_mul(yaw_quat(yaw), [np.cos(tilt / 2), 0.0, np.sin(tilt / 2), 0.0])
        b.cylinder([hx, hy, 0.75], 0.018, 0.42, material=m["dark"], quat=q)
        gx, gy = rot(-0.88, 0.0)
        b.box([gx, gy, 1.14], [0.03, 0.14, 0.02], yaw, material=m["red"])

    def fence(self, x, y0, y1, height=1.4):
        """Safety fence behind the robot cell: yellow posts, dark mesh panels, caution sign."""
        b, m = self.b, self.m
        posts = np.linspace(y0, y1, 4)
        for y in posts:
            b.box([x, y, height / 2], [0.025, 0.025, height / 2], material=m["yellow"])
        for ya, yb in zip(posts[:-1], posts[1:]):
            ym, half = (ya + yb) / 2, (yb - ya) / 2 - 0.025
            b.box([x, ym, 0.15 + (height - 0.25) / 2], [0.004, half, (height - 0.25) / 2], material=m["mesh"])
            b.box([x, ym, height - 0.03], [0.015, half, 0.02], material=m["yellow"])
        # caution sign facing +x (towards the robot and the cameras)
        b.sign([x + 0.01, (posts[1] + posts[2]) / 2, 1.0], 0.2, 0.05, "+x", m["sign_caution"])

    def inspection_gantry(self, x_post, x_tip, y, top):
        """Cantilever frame over the inspection station carrying a light bar and two signs.
        The posts stand at x_post, behind the robot's front camera and outside the inspection
        cameras' cone; the arms run above the front camera's view to x_tip."""
        b, m = self.b, self.m
        for py in (-y, y):
            b.box([x_post, py, top / 2], [0.03, 0.03, top / 2], material=m["steel"])
            b.box([(x_post + x_tip) / 2, py, top], [(x_post - x_tip) / 2 + 0.03, 0.025, 0.025], material=m["steel"])
            b.sign([0.2, py + np.sign(py) * 0.03, top - 0.07], 0.19, 0.048, "+y" if py > 0 else "-y",
                   m["sign_inspection"])
        for px in (x_post, x_tip):
            b.box([px, 0.0, top], [0.025, y + 0.025, 0.025], material=m["steel"])
        b.box([0.2, 0.0, top - 0.035], [0.02, y - 0.03, 0.008], material=m["light_strip"])

    def cabinet(self, x, y):
        """Electrical cabinet with an HMI screen and an e-stop, and a stack light on top."""
        b, m = self.b, self.m
        b.box([x, y, 0.8], [0.3, 0.17, 0.8], material=m["cabinet"])
        b.box([x, y - 0.171, 0.8], [0.002, 0.001, 0.75], material=m["dark"])          # door seam
        b.sign([x - 0.12, y - 0.172, 1.25], 0.12, 0.08, "-y", m["screen"])
        b.cylinder([x + 0.16, y - 0.18, 1.25], 0.03, 0.012, material=m["red"],
                   quat=[0.7071, 0.7071, 0, 0])
        b.cylinder([x + 0.2, y, 1.65], 0.012, 0.05, material=m["steel"])
        for k, lamp in enumerate(("lamp_g", "lamp_y", "lamp_r")):
            b.cylinder([x + 0.2, y, 1.73 + 0.06 * k], 0.035, 0.028, material=m[lamp])

    def packing_sign(self, x, y, z=1.25, half_w=0.32):
        """Sign on a post next to the packing carton, facing +x."""
        b, m = self.b, self.m
        b.box([x - 0.02, y, z / 2], [0.02, 0.02, z / 2], material=m["steel"])
        b.sign([x, y, z + 0.08], half_w, half_w / 4, "+x", m["sign_packing"])

    def drums(self, xys):
        for x, y in xys:
            self.b.cylinder([x, y, 0.44], 0.29, 0.44, material=self.m["drum"])
            self.b.cylinder([x, y, 0.885], 0.27, 0.006, material=self.m["dark"])

    def floor_markings(self):
        """Hazard tape around the robot cell and yellow walkway lines."""
        b, m = self.b, self.m
        x0, x1, y0, y1, w = -1.08, -0.15, -0.62, 0.9, 0.04
        b.box([(x0 + x1) / 2, y1, 0.003], [(x1 - x0) / 2, w, 0.002], material=m["hazard"])
        b.box([x0, (y0 + y1) / 2, 0.003], [(y1 - y0) / 2, w, 0.002], yaw=np.pi / 2, material=m["hazard"])
        for y in (1.25, -1.95):
            b.box([0.5, y, 0.002], [1.9, 0.03, 0.001], material=m["yellow"])


def _quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]

