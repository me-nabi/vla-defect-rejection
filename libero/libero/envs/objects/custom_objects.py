import os
import re
import numpy as np
from robosuite.models.objects import MujocoXMLObject
import pathlib

absolute_path = pathlib.Path(__file__).parent.parent.parent.absolute()

from libero.libero.envs.base_object import register_object


class CustomBaseObject(MujocoXMLObject):
    def __init__(self, name, obj_name, obj_dir):
        super().__init__(
            os.path.join(
                str(absolute_path),
                f"assets/custom_objects/{obj_dir}/{obj_name}.xml",
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
class RejectBin(CustomBaseObject):
    def __init__(self, name="reject_bin", obj_name="reject_bin"):
        super().__init__(name, obj_name, obj_dir="reject_bin")
        self.rotation = (0, 0)
        self.rotation_axis = "z"
