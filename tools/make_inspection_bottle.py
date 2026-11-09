"""Generate the assets of the `inspection_bottle` object, one folder per bottle design.

    python tools/make_inspection_bottle.py

Writes libero/libero/assets/conveyor_line/inspection_bottle/<design>/ for every design in
libero/libero/envs/bottle_shapes.py:
    glass.obj        lathed body (glass or PET) with shoulder, neck and an open lip
    label.obj        thin label band with texture coordinates
    label.png        generic product label (no real brand)
    liquid_lo.obj    liquid from the base up to `liquid_split` (follows waist and ribs)
    dent*.obj        PET only: the body with a dent, and dent_liquid*.obj its liquid
    inspection_bottle.xml

The defect geoms (crack segments, chip flakes, a floating particle, dented bodies) and the
harmless look-alikes (air bubbles, condensation droplets) are in the XML but invisible.
The environment shows, moves and resizes them at reset (libero/libero/envs/inspection_defects.py),
along with colours, the cap and the liquid level (the liquid above `liquid_split` is a
cylinder that gets resized). Physics: one collision cylinder per design and an explicit
inertia, so grasping doesn't depend on the visual meshes.
"""
import os
import sys

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
from libero.libero.envs.bottle_shapes import SHAPES  # noqa: E402

OUT = os.path.join(REPO, "libero", "libero", "assets", "conveyor_line", "inspection_bottle")
N_SEG = 64  # segments around the axis
LIQUID_Z0 = 0.004

N_CRACK_SEGMENTS = 8
N_CHIP_FLAKES = 3
N_BUBBLES = 6       # harmless air bubbles in the juice (not a defect)
N_DROPS = 12        # harmless condensation droplets on the glass (not a defect)


def body_profile(s, z_end, dz=0.001):
    zs = np.arange(0.003, z_end, dz)
    return [(float(s.body_radius(z)), float(z)) for z in zs] + [(float(s.body_radius(z_end)), z_end)]


def shoulder_profile(s, n=12):
    """Smooth (cosine) taper from the body radius down to the neck."""
    t = np.linspace(0, 1, n)
    r = s.r_neck + (s.r_body - s.r_neck) * 0.5 * (1 + np.cos(np.pi * t))
    z = s.z_shoulder + (s.z_neck - s.z_shoulder) * t
    return list(zip(r, z))


def glass_profile(s):
    """(r, z) outline, bottom centre -> outside -> lip -> down the bore -> bore centre."""
    pts = [(0.0, 0.0), (s.r_body * 0.886, 0.0)]
    pts += body_profile(s, s.z_shoulder)
    pts += shoulder_profile(s)[1:]
    pts += [(s.r_neck, s.z_lip0), (s.r_lip, s.z_lip0 + 0.0005), (s.r_lip, s.z_top - 0.0005),
            (s.r_lip - 0.0008, s.z_top), (s.r_bore, s.z_top), (s.r_bore, s.z_bore), (0.0, s.z_bore)]
    return pts


def liquid_profile(s, z_top, closed):
    """Inside outline from the liquid bottom up to z_top; closed adds a flat liquid surface."""
    zs = np.arange(LIQUID_Z0 + 0.001, z_top, 0.001)
    pts = [(0.0, LIQUID_Z0), (float(s.inner_radius(LIQUID_Z0)) - 0.0005, LIQUID_Z0)]
    pts += [(float(s.inner_radius(z)) - 0.0002, float(z)) for z in zs]
    pts += [(float(s.inner_radius(z_top)) - 0.0002, z_top)]
    if closed:
        pts.append((0.0, z_top))
    return pts


def lathe(profile, closed_loop=False, uv_fn=None):
    """Revolve a profile around z. Returns (vertices, texcoords, faces), outward winding.

    Points with r == 0 become a single vertex (fan). Each ring gets N_SEG + 1 vertices
    (seam duplicated) so texture coordinates can wrap.
    """
    verts, uvs, rings = [], [], []
    for k, (r, z) in enumerate(profile):
        if r == 0.0:
            rings.append([len(verts)])
            verts.append((0.0, 0.0, z))
            uvs.append((0.0, 0.0))
            continue
        ring = []
        for j in range(N_SEG + 1):
            phi = 2 * np.pi * j / N_SEG
            ring.append(len(verts))
            verts.append((r * np.cos(phi), r * np.sin(phi), z))
            uvs.append(uv_fn(k, j, r, z) if uv_fn else (j / N_SEG, 0.0))
        rings.append(ring)
    faces = []
    pairs = list(zip(rings[:-1], rings[1:]))
    if closed_loop:
        pairs.append((rings[-1], rings[0]))
    for a, b in pairs:
        for j in range(N_SEG):
            if len(a) == 1:
                faces.append((a[0], b[j + 1], b[j]))
            elif len(b) == 1:
                faces.append((a[j], a[j + 1], b[0]))
            else:
                faces.append((a[j], a[j + 1], b[j + 1]))
                faces.append((a[j], b[j + 1], b[j]))
    return verts, uvs, faces


def dent(verts, z0, depth, sigma_phi, sigma_z, phi0=0.0):
    """Push vertices radially inwards with a Gaussian dent centred at angle phi0, height z0."""
    out = []
    for x, y, z in verts:
        r = np.hypot(x, y)
        if r < 1e-9:
            out.append((x, y, z))
            continue
        phi = np.arctan2(y, x)
        dphi = (phi - phi0 + np.pi) % (2 * np.pi) - np.pi
        push = depth * np.exp(-(dphi / sigma_phi) ** 2 - ((z - z0) / sigma_z) ** 2)
        k = max(r - push, 0.3 * r) / r
        out.append((x * k, y * k, z))
    return out


def write_obj(path, verts, uvs, faces):
    with open(path, "w") as f:
        f.write("# generated by tools/make_inspection_bottle.py\n")
        for v in verts:
            f.write("v %.6f %.6f %.6f\n" % tuple(v))
        for t in uvs:
            f.write("vt %.6f %.6f\n" % tuple(t))
        for a, b, c in faces:
            f.write(f"f {a + 1}/{a + 1} {b + 1}/{b + 1} {c + 1}/{c + 1}\n")


def make_label_png(path, brand, product, seed):
    """A generic drinks label: paper, coloured bands, product text and a barcode. White-ish
    so the environment can tint it."""
    w, h = 1024, 176
    img = Image.new("RGB", (w, h), (246, 244, 238))
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, w, 22], fill=(200, 200, 205))
    d.rectangle([0, h - 22, w, h], fill=(200, 200, 205))
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 64)
        small = ImageFont.truetype("DejaVuSans.ttf", 30)
    except OSError:
        font = small = ImageFont.load_default()
    for x0 in (40, 552):  # printed twice around the bottle
        d.ellipse([x0, 42, x0 + 92, 134], fill=(150, 150, 150), outline=(90, 90, 90), width=4)
        d.text((x0 + 116, 40), brand, font=font, fill=(45, 45, 50))
        d.text((x0 + 120, 112), product, font=small, fill=(80, 80, 85))
    rng = np.random.default_rng(seed)
    x = 430
    while x < 500:  # barcode
        bw = int(rng.integers(1, 4))
        d.rectangle([x, 50, x + bw, 128], fill=(20, 20, 20))
        x += bw + int(rng.integers(1, 4))
    img.save(path)


XML = """<mujoco model="inspection_bottle">
  <!-- Generated by tools/make_inspection_bottle.py ({name}); edit the generator, not this file. -->
  <asset>
    <mesh name="glass" file="glass.obj"/>
    <mesh name="label" file="label.obj"/>
    <mesh name="liquid_lo" file="liquid_lo.obj" inertia="shell"/>
{dent_meshes}
    <texture name="label_tex" type="2d" file="label.png"/>
    <material name="glass_mat" rgba="0.82 0.93 0.90 0.30" specular="0.9" shininess="0.95" reflectance="0.1"/>
    <material name="label_mat" texture="label_tex" rgba="1 1 1 1" specular="0.1" shininess="0.2"/>
    <material name="liquid_mat" rgba="0.96 0.55 0.10 0.50" specular="0.4" shininess="0.6"/>
    <material name="cap_mat" rgba="0.12 0.32 0.72 1" specular="0.5" shininess="0.6"/>
    <material name="flaw_mat" rgba="0.95 0.97 1.0 0.85" specular="1" shininess="1" emission="0.25"/>
    <material name="particle_mat" rgba="0.10 0.08 0.06 1" specular="0.2" shininess="0.2"/>
    <material name="bubble_mat" rgba="0.97 0.97 0.95 0.6" specular="0.9" shininess="0.9"/>
    <material name="drop_mat" rgba="0.90 0.95 1.0 0.45" specular="1" shininess="1" reflectance="0.2"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <!-- Inertia of a solid cylinder of the bottle's size and mass -->
        <inertial pos="0 0 {com_z}" mass="{mass}" diaginertia="{ixx} {ixx} {izz}"/>
        <geom name="glass" type="mesh" mesh="glass" material="glass_mat" contype="0" conaffinity="0" group="1"/>
        <geom name="label" type="mesh" mesh="label" material="label_mat" contype="0" conaffinity="0" group="1"/>
        <geom name="liquid_lo" type="mesh" mesh="liquid_lo" material="liquid_mat" contype="0" conaffinity="0" group="1"/>
        <geom name="liquid_hi" type="cylinder" size="{liq_r} 0.01" pos="0 0 {liq_z}" material="liquid_mat" contype="0" conaffinity="0" group="1"/>
        <geom name="cap" type="cylinder" size="{cap_r} {cap_hh}" pos="0 0 {cap_z}" material="cap_mat" contype="0" conaffinity="0" group="1"/>
        <!-- Defect geoms: hidden (alpha 0) until the environment enables them -->
        <geom name="particle" type="sphere" size="0.0015" pos="0 0 0.05" material="particle_mat" rgba="0.1 0.08 0.06 0" contype="0" conaffinity="0" group="1"/>
{flaws}
        <!-- Collision geometry -->
        <!-- Default (stable) contact softness: a stiffer solref than ~2x the 2 ms timestep
             makes bottle-on-bottle contacts explode. Friction is set low at runtime (the
             environment drives bottles on the belt). -->
        <geom name="collision" type="cylinder" size="{r_body} {half_h}" pos="0 0 {half_h}"
              friction="1.0 0.005 0.0001" group="0" rgba="0.8 0.8 0.8 0"/>
      </body>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 -0.005" name="bottom_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="0 0 {height}" name="top_site"/>
      <site rgba="0 0 0 0" size="0.005" pos="{site_r} {site_r} 0" name="horizontal_radius_site"/>
    </body>
  </worldbody>
</mujoco>
"""


def build(s, seed):
    out = os.path.join(OUT, s.name)
    os.makedirs(out, exist_ok=True)
    glass = lathe(glass_profile(s))
    write_obj(os.path.join(out, "glass.obj"), *glass)
    write_obj(os.path.join(out, "liquid_lo.obj"), *lathe(liquid_profile(s, s.liquid_split, closed=False)))

    z0, z1 = s.label_z
    r_in = s.label_radius() + 0.00005
    r_out = r_in + 0.00017
    label_profile = [(r_in, z0), (r_out, z0), (r_out, z1), (r_in, z1)]

    def label_uv(k, j, r, z):
        if r == r_out:  # printed outer face
            return (j / N_SEG, (z - z0) / (z1 - z0))
        return (j / N_SEG, 0.0)

    write_obj(os.path.join(out, "label.obj"), *lathe(label_profile, closed_loop=True, uv_fn=label_uv))
    make_label_png(os.path.join(out, "label.png"), *s.label_text, seed=seed)

    flaw = ('        <geom name="{n}" type="{t}" size="{s}" material="flaw_mat" rgba="0.95 0.97 1 0" '
            'contype="0" conaffinity="0" group="1"/>')
    flaws = [flaw.format(n=f"crack{i}", t="box", s="0.003 0.0004 0.0002") for i in range(N_CRACK_SEGMENTS)]
    flaws += [flaw.format(n=f"chip{i}", t="ellipsoid", s="0.002 0.002 0.001") for i in range(N_CHIP_FLAKES)]
    benign = ('        <geom name="{n}" type="{t}" size="{s}" material="{m}" rgba="1 1 1 0" '
              'contype="0" conaffinity="0" group="1"/>')
    flaws += [benign.format(n=f"bubble{i}", t="sphere", s="0.0008", m="bubble_mat") for i in range(N_BUBBLES)]
    flaws += [benign.format(n=f"drop{i}", t="ellipsoid", s="0.0007 0.0007 0.0003", m="drop_mat") for i in range(N_DROPS)]

    dent_meshes = []
    for k, (_, zc, depth, s_phi, s_z) in enumerate(s.dents):
        write_obj(os.path.join(out, f"dent{k}.obj"), dent(glass[0], zc, depth, s_phi, s_z), glass[1], glass[2])
        liq = lathe(liquid_profile(s, s.dent_fill, closed=True))
        write_obj(os.path.join(out, f"dent_liquid{k}.obj"), dent(liq[0], zc, depth, s_phi, s_z), liq[1], liq[2])
        dent_meshes += [f'    <mesh name="dent{k}" file="dent{k}.obj"/>',
                        f'    <mesh name="dent_liquid{k}" file="dent_liquid{k}.obj"/>']
        flaws += [f'        <geom name="dent{k}" type="mesh" mesh="dent{k}" material="glass_mat" rgba="1 1 1 0" '
                  'contype="0" conaffinity="0" group="1"/>',
                  f'        <geom name="dent_liquid{k}" type="mesh" mesh="dent_liquid{k}" material="liquid_mat" '
                  'rgba="1 1 1 0" contype="0" conaffinity="0" group="1"/>']

    h = s.z_top
    m, r = s.mass, s.r_body
    xml = XML.format(
        name=s.name, dent_meshes="\n".join(dent_meshes), flaws="\n".join(flaws),
        com_z=f"{h / 2:.4f}", mass=m, ixx=f"{m * (3 * r * r + h * h) / 12:.3e}", izz=f"{m * r * r / 2:.3e}",
        liq_r=f"{s.liquid_r_hi:.4f}", liq_z=f"{s.liquid_split + 0.01:.4f}",
        cap_r=s.cap_r, cap_hh=s.cap_half_h, cap_z=f"{s.cap_z:.4f}",
        r_body=r, half_h=f"{h / 2:.4f}", height=f"{s.height:.4f}", site_r=f"{0.9 * r:.4f}",
    )
    with open(os.path.join(out, "inspection_bottle.xml"), "w") as f:
        f.write(xml)
    print("wrote", out)


def main():
    for k, s in enumerate(SHAPES.values()):
        build(s, seed=7 + k)


if __name__ == "__main__":
    main()
