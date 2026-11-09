import os
import re
import numpy as np
from robosuite.models.objects import MujocoXMLObject
import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from libero.libero.envs.base_object import register_object
from libero.libero.envs.bottle_shapes import DEFAULT_WEIGHTS, SHAPES

CONVEYOR_ASSETS = "conveyor_line"   # assets/conveyor_line/: bottles, reject box, warehouse textures


class CustomBaseObject(MujocoXMLObject):
    def __init__(self, name, obj_name, obj_dir, assets_dir="custom_objects"):
        super().__init__(
            os.path.join(
                str(absolute_path),
                f"assets/{assets_dir}/{obj_dir}/{obj_name}.xml",
            ),
            name=name,
            joints=[dict(type="free", damping="0.0005")],
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z0-9])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (np.pi / 2, np.pi / 2)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


class CustomFixtureObject(MujocoXMLObject):
    def __init__(self, name, obj_name, obj_dir):
        super().__init__(
            os.path.join(
                str(absolute_path),
                f"assets/custom_objects/{obj_dir}/{obj_name}.xml",
            ),
            name=name,
            joints=None,
            obj_type="all",
            duplicate_collision_geoms=False,
        )
        self.category_name = "_".join(
            re.sub(r"([A-Z0-9])", r" \1", self.__class__.__name__).split()
        ).lower()
        self.rotation = (0, 0)
        self.rotation_axis = "z"
        self.object_properties = {"vis_site_names": {}}


@register_object
class GlassBottle(CustomBaseObject):
    def __init__(self, name="glass_bottle", obj_name="glass_bottle"):
        super().__init__(name, obj_name, obj_dir="glass_bottle")
        self.rotation = (0, 0)
        self.rotation_axis = "z"


@register_object
class GoodBottle(CustomBaseObject):
    """Non-defective bottle: same geometry as GlassBottle, different appearance."""

    def __init__(self, name="good_bottle", obj_name="good_bottle"):
        super().__init__(name, obj_name, obj_dir="good_bottle")
        self.rotation = (0, 0)
        self.rotation_axis = "z"


@register_object
class InspectionBottle(CustomBaseObject):
    """Drinks bottle for the inspection line. Each instance is one of the designs in
    libero/libero/envs/bottle_shapes.py (glass 250/500/330 ml, PET 330 ml), drawn at random
    with VARIANT_WEIGHTS when the object is created, i.e. on every hard reset. The environment
    then decides which bottles are defective and how (libero/libero/envs/inspection_defects.py).
    Assets come from tools/make_inspection_bottle.py."""

    VARIANT_WEIGHTS = dict(DEFAULT_WEIGHTS)   # the environment may override this

    def __init__(self, name="inspection_bottle", obj_name="inspection_bottle", variant=None):
        if variant is None:
            names = sorted(self.VARIANT_WEIGHTS)
            w = np.array([self.VARIANT_WEIGHTS[n] for n in names], float)
            variant = names[int(np.random.choice(len(names), p=w / w.sum()))]
        self.shape = SHAPES[variant]
        super().__init__(name, obj_name, obj_dir=f"inspection_bottle/{variant}", assets_dir=CONVEYOR_ASSETS)
        # Keep 0: LIBERO's table sampler returns (x, y, z, w) quaternions that are then used as
        # (w, x, y, z), so any other angle tips the bottle over. The environment randomizes
        # the yaw itself (Conveyor_Belt_Manipulation.randomize_bottles).
        self.rotation = (0, 0)
        self.rotation_axis = "z"


@register_object
class RejectBin(CustomBaseObject):
    def __init__(self, name="reject_bin", obj_name="reject_bin"):
        super().__init__(name, obj_name, obj_dir="reject_bin", assets_dir=CONVEYOR_ASSETS)
        self.rotation = (0, 0)
        self.rotation_axis = "z"
