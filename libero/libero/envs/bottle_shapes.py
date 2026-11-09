"""Bottle designs for the inspection line, shared by the asset generator
(tools/make_inspection_bottle.py) and the appearance/defect code (inspection_defects.py).

Every design is a body of revolution described by a few dimensions. `outer_radius(z)` gives
the outside radius at height z (bottle frame: base at z = 0), so defects, droplets and the
liquid can follow any outline, including the pinched waist of the plastic bottle.

    glass_250   the original 250 ml glass juice bottle
    glass_500   tall 500 ml glass bottle (the big one)
    glass_330   short, wide 330 ml glass bottle
    pet_330     330 ml plastic (PET) bottle with a pinched grip waist and ribs; PET bottles
                can arrive dented, glass ones can be cracked or chipped
"""
from dataclasses import dataclass, field
from typing import Optional, Tuple

import numpy as np

GLASS_DEFECTS = ("crack", "chip", "missing_cap", "tilted_cap", "low_fill", "particle")
PET_DEFECTS = ("missing_cap", "tilted_cap", "low_fill", "particle", "dented")
ALL_DEFECTS = ("crack", "chip", "missing_cap", "tilted_cap", "low_fill", "particle", "dented")


@dataclass(frozen=True)
class BottleShape:
    name: str
    material: str                      # "glass" or "pet"
    volume_ml: int
    r_body: float
    z_shoulder: float                  # body ends, shoulder taper begins
    z_neck: float                      # taper ends at the neck radius
    r_neck: float
    r_lip: float
    r_bore: float
    z_lip0: float
    z_top: float
    z_bore: float
    label_z: Tuple[float, float]
    wall: float                        # wall thickness (liquid radius = outer radius - wall)
    liquid_split: float                # liquid mesh below, resizable cylinder above (seam hidden behind the label where possible)
    normal_fill: Tuple[float, float]
    low_fill: Tuple[float, float]
    cap_r: float
    cap_half_h: float
    mass: float
    crack_bands: Tuple[Tuple[float, float], ...] = ()
    chip_zones: Tuple[Tuple[float, float], ...] = ()
    particle_zones: Tuple[Tuple[float, Optional[float]], ...] = ()   # (lo, hi); hi None = up to the fill level
    waist: Optional[Tuple[float, float, float, float]] = None        # (z0, z1, r_waist, transition length)
    ribs: Tuple[float, ...] = ()                                     # heights of small grip ribs
    rib_depth: float = 0.0012
    dents: Tuple[Tuple[str, float, float, float, float], ...] = ()   # (name, z, depth, sigma_phi, sigma_z)
    dent_fill: float = 0.0
    label_text: Tuple[str, str] = ("FRESCO", "fruit juice")

    # ---- derived ----------------------------------------------------------------------
    @property
    def defects(self):
        return PET_DEFECTS if self.material == "pet" else GLASS_DEFECTS

    @property
    def cap_z(self):
        return self.z_top + 0.0005

    @property
    def height(self):
        """Overall height with the cap on."""
        return self.cap_z + self.cap_half_h

    @property
    def body_top(self):
        return self.z_shoulder

    def waist_dip(self, z):
        if self.waist is None:
            return np.zeros_like(np.asarray(z, float))
        z = np.asarray(z, float)
        z0, z1, r_w, tr = self.waist
        depth = self.r_body - r_w
        d = np.zeros_like(z)
        rise = (z > z0 - tr) & (z < z0)
        d[rise] = depth * 0.5 * (1 - np.cos(np.pi * (z[rise] - (z0 - tr)) / tr))
        d[(z >= z0) & (z <= z1)] = depth
        fall = (z > z1) & (z < z1 + tr)
        d[fall] = depth * 0.5 * (1 + np.cos(np.pi * (z[fall] - z1) / tr))
        return d

    def rib_dip(self, z):
        z = np.asarray(z, float)
        d = np.zeros_like(z)
        for zr in self.ribs:
            d += self.rib_depth * np.exp(-((z - zr) / 0.0012) ** 2)
        return d

    def body_radius(self, z):
        """Outside radius on the straight/waisted body (base rounding and shoulder excluded)."""
        return self.r_body - self.waist_dip(z) - self.rib_dip(z)

    def outer_radius(self, z):
        """Outside radius anywhere from the base to the top of the neck."""
        z = np.asarray(z, float)
        r = self.body_radius(z)
        t = np.clip((z - self.z_shoulder) / (self.z_neck - self.z_shoulder), 0, 1)
        shoulder = self.r_neck + (self.r_body - self.r_neck) * 0.5 * (1 + np.cos(np.pi * t))
        r = np.where(z > self.z_shoulder, shoulder, r)
        return np.where(z > self.z_neck, self.r_neck, r)

    def inner_radius(self, z):
        return self.outer_radius(z) - self.wall

    @property
    def liquid_r_hi(self):
        """Radius of the resizable liquid cylinder: the narrowest inside radius it spans."""
        zs = np.linspace(self.liquid_split, self.normal_fill[1], 50)
        return float(self.inner_radius(zs).min()) - 0.0002

    def label_radius(self):
        z0, z1 = self.label_z
        return float(self.outer_radius(np.linspace(z0, z1, 20)).max())


SHAPES = {
    "glass_250": BottleShape(
        name="glass_250", material="glass", volume_ml=250,
        r_body=0.022, z_shoulder=0.068, z_neck=0.088, r_neck=0.0105, r_lip=0.0115, r_bore=0.0078,
        z_lip0=0.0995, z_top=0.103, z_bore=0.094, label_z=(0.010, 0.034), wall=0.0025,
        liquid_split=0.030, normal_fill=(0.060, 0.067), low_fill=(0.045, 0.058),
        cap_r=0.0122, cap_half_h=0.0045, mass=0.134,
        crack_bands=((0.036, 0.066), (0.002, 0.009)), chip_zones=((0.0015, 0.004), (0.056, 0.065)),
        particle_zones=((0.038, None),), label_text=("FRESCO", "fruit juice  250 ml"),
    ),
    "glass_500": BottleShape(
        name="glass_500", material="glass", volume_ml=500,
        r_body=0.0275, z_shoulder=0.100, z_neck=0.128, r_neck=0.012, r_lip=0.013, r_bore=0.009,
        z_lip0=0.1445, z_top=0.149, z_bore=0.138, label_z=(0.014, 0.050), wall=0.003,
        liquid_split=0.046, normal_fill=(0.088, 0.097), low_fill=(0.064, 0.083),
        cap_r=0.0138, cap_half_h=0.005, mass=0.26,
        crack_bands=((0.052, 0.097), (0.002, 0.011)), chip_zones=((0.0015, 0.005), (0.083, 0.096)),
        particle_zones=((0.054, None),), label_text=("FRESCO", "fruit juice  500 ml"),
    ),
    "glass_330": BottleShape(
        name="glass_330", material="glass", volume_ml=330,
        r_body=0.029, z_shoulder=0.056, z_neck=0.074, r_neck=0.0115, r_lip=0.0125, r_bore=0.0085,
        z_lip0=0.0835, z_top=0.087, z_bore=0.078, label_z=(0.010, 0.032), wall=0.003,
        liquid_split=0.028, normal_fill=(0.049, 0.054), low_fill=(0.036, 0.045),
        cap_r=0.0133, cap_half_h=0.0045, mass=0.20,
        crack_bands=((0.034, 0.054), (0.002, 0.009)), chip_zones=((0.0015, 0.004), (0.045, 0.053)),
        particle_zones=((0.036, None),), label_text=("FRESCO", "nectar  330 ml"),
    ),
    "pet_330": BottleShape(
        name="pet_330", material="pet", volume_ml=330,
        r_body=0.028, z_shoulder=0.118, z_neck=0.140, r_neck=0.0125, r_lip=0.0135, r_bore=0.0105,
        z_lip0=0.150, z_top=0.156, z_bore=0.148, label_z=(0.050, 0.080), wall=0.0008,
        liquid_split=0.093, normal_fill=(0.106, 0.113), low_fill=(0.096, 0.1035),
        cap_r=0.0148, cap_half_h=0.0065, mass=0.22,
        particle_zones=((0.008, 0.036), (0.095, None)),
        waist=(0.050, 0.080, 0.022, 0.012), ribs=(0.016, 0.026),
        dents=(("dent_low", 0.024, 0.006, 0.40, 0.010),
               ("dent_high", 0.104, 0.005, 0.45, 0.008),
               ("crease", 0.100, 0.0035, 0.90, 0.0035)),
        dent_fill=0.110, label_text=("FRESCO", "sparkling  330 ml"),
    ),
}

# How often each design appears on the line.
DEFAULT_WEIGHTS = {"glass_250": 0.3, "glass_500": 0.25, "glass_330": 0.2, "pet_330": 0.25}
