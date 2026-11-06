import os

import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import (
    array_to_string,
    string_to_array,
    xml_path_completion,
)

from libero.libero.envs.arenas.style import get_texture_filename

DEFAULT_CONVEYOR_XML = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "assets", "scenes", "libero_conveyor_belt.xml",
))


class ConveyorBeltArena(Arena):
    def __init__(
        self,
        table_full_size=(1.0, 0.4, 0.05),
        table_friction=(0.6, 0.005, 0.0001),
        table_offset=(0, 0, 0.8),
        has_legs=True,
        xml=DEFAULT_CONVEYOR_XML,
        floor_style="light-gray",
        wall_style="light-gray-plaster",
        belt_extension=0.0,
    ):
        """
        Args:
            belt_extension (float): extra belt length (m) added on the upstream (+x)
                end only. The belt body origin, robot base and BDDL region
                coordinates are unchanged, so existing tasks stay identical when 0.
        """
        super().__init__(xml_path_completion(xml))
        self.belt_extension = float(belt_extension)

        self.table_full_size = np.array(table_full_size)
        self.table_half_size = self.table_full_size / 2
        self.table_friction = table_friction
        self.table_offset = table_offset
        self.center_pos = (
            self.bottom_pos
            + np.array([0, 0, -self.table_half_size[2]])
            + self.table_offset
        )

        self.table_body = self.worldbody.find("./body[@name='table']")
        self.table_collision = self.table_body.find("./geom[@name='table_collision']")
        self.table_visual = self.table_body.find("./geom[@name='table_visual']")
        self.table_top = self.table_body.find("./site[@name='table_top']")
        # Decorative geoms that run along the belt. They are optional (scene
        # variants include different subsets) and are resized with the belt so
        # the visuals always match the collision surface.
        self.belt_full_length_geoms = self._find_geoms(
            ["rail_left", "rail_right", "belt_stripe_l", "belt_stripe_r"]
        )
        self.belt_between_legs_geoms = self._find_geoms(["brace_left", "brace_right"])

        self.has_legs = has_legs
        self.table_legs_visual = [
            self.table_body.find("./geom[@name='table_leg1_visual']"),
            self.table_body.find("./geom[@name='table_leg2_visual']"),
            self.table_body.find("./geom[@name='table_leg3_visual']"),
            self.table_body.find("./geom[@name='table_leg4_visual']"),
        ]

        self.configure_location()

        texplane = self.asset.find("./texture[@name='texplane']")
        plane_file = texplane.get("file")
        plane_file = "/".join(
            plane_file.split("/")[:-1]
            + [get_texture_filename(type="floor", style=floor_style)]
        )
        texplane.set("file", plane_file)

        texwall = self.asset.find("./texture[@name='tex-wall']")
        wall_file = texwall.get("file")
        wall_file = "/".join(
            wall_file.split("/")[:-1]
            + [get_texture_filename(type="wall", style=wall_style)]
        )
        texwall.set("file", wall_file)

    def configure_location(self):
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.table_body.set("pos", array_to_string(self.center_pos))

        # Belt spans x in [-half_x, half_x + belt_extension] in the body frame.
        belt_half_x = self.table_half_size[0] + self.belt_extension / 2.0
        belt_center_x = self.belt_extension / 2.0
        belt_half_size = np.array(
            [belt_half_x, self.table_half_size[1], self.table_half_size[2]]
        )
        for geom in (self.table_collision, self.table_visual):
            geom.set("size", array_to_string(belt_half_size))
            geom.set("pos", array_to_string([belt_center_x, 0, 0]))
        self.table_collision.set("friction", array_to_string(self.table_friction))

        for geom in self.belt_full_length_geoms:
            self._stretch_along_belt(geom, belt_center_x, belt_half_x)

        self.table_top.set(
            "pos", array_to_string(np.array([0, 0, self.table_half_size[2]]))
        )

        leg_inset = 0.1  # legs sit this far in from each belt end
        if not self.has_legs:
            for leg in self.table_legs_visual:
                leg.set("rgba", array_to_string([1, 0, 0, 0]))
                leg.set("size", array_to_string([0.0001, 0.0001]))
            for geom in self.belt_between_legs_geoms:
                geom.set("rgba", array_to_string([1, 0, 0, 0]))
        else:
            leg_half_height = (self.table_offset[2] - self.table_half_size[2]) / 2.0
            for leg in self.table_legs_visual:
                # Keep each leg's side (+/-x) and y offset from the scene XML.
                xml_pos = string_to_array(leg.get("pos"))
                side = 1.0 if xml_pos[0] >= 0 else -1.0
                x = belt_center_x + side * max(belt_half_x - leg_inset, 0.0)
                leg.set("pos", array_to_string([x, xml_pos[1], -leg_half_height]))
                leg.set("size", array_to_string([0.025, leg_half_height]))
            for geom in self.belt_between_legs_geoms:
                self._stretch_along_belt(
                    geom, belt_center_x, max(belt_half_x - leg_inset, 0.0)
                )

    def _find_geoms(self, names):
        geoms = [self.table_body.find(f"./geom[@name='{n}']") for n in names]
        return [g for g in geoms if g is not None]

    @staticmethod
    def _stretch_along_belt(geom, center_x, half_x):
        pos = string_to_array(geom.get("pos"))
        size = string_to_array(geom.get("size"))
        pos[0], size[0] = center_x, half_x
        geom.set("pos", array_to_string(pos))
        geom.set("size", array_to_string(size))

    @property
    def table_top_abs(self):
        return string_to_array(self.floor.get("pos")) + self.table_offset
