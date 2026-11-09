import os
from dataclasses import dataclass

import numpy as np

from robosuite.models.arenas import Arena
from robosuite.utils.mjcf_utils import (
    array_to_string,
    new_element,
    string_to_array,
    xml_path_completion,
)

from libero.libero.envs.arenas.style import get_texture_filename
from libero.libero.envs.arenas.warehouse_props import TEX_DIR, WarehouseProps, facing_quat

DEFAULT_CONVEYOR_XML = os.path.normpath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "..", "assets", "scenes", "libero_conveyor_belt.xml",
))


@dataclass(frozen=True)
class BeltExit:
    """Where good bottles go after the robot: the main belt ends before the robot's
    pedestal, a 90-degree bend turns the line away from the robot and the reject bin
    (towards -y), and a straight exit belt carries the bottles into a crate.

    All values are world coordinates in metres (the belt body sits at x = y = 0).
    """
    x_end: float = -0.27        # downstream end of the main belt
    radius: float = 0.18        # centreline radius of the bend
    width: float = 0.14         # width of the bend and the exit belt (guide rails at the edges)
    length: float = 1.0         # straight exit belt after the bend
    crate_half: float = 0.15    # inner half-size of the crate at the end
    crate_drop: float = 0.16    # crate floor this far below the belt surface

    @property
    def center(self):
        """Centre of the bend; the centreline runs from (x_end, 0) to (x_end - radius, -radius)."""
        return np.array([self.x_end, -self.radius])

    @property
    def exit_x(self):
        return self.x_end - self.radius

    @property
    def exit_y_end(self):
        return -self.radius - self.length


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
        belt_exit=None,
        warehouse_props=None,
    ):
        """
        Args:
            belt_extension (float): extra belt length (m) added on the upstream (+x)
                end only. The belt body origin, robot base and BDDL region
                coordinates are unchanged, so existing tasks stay identical when 0.
            belt_exit (BeltExit or None): if given, the main belt ends at
                belt_exit.x_end and continues through a bend and an exit belt into a
                crate, instead of running into the robot's pedestal.
            warehouse_props (bool or None): add visual-only warehouse dressing (racking,
                pallets, fence, signs, ...). None means "with the belt exit".
        """
        super().__init__(xml_path_completion(xml))
        self.belt_extension = float(belt_extension)
        self.belt_exit = belt_exit
        self.warehouse_props = belt_exit is not None if warehouse_props is None else bool(warehouse_props)

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

        if self.warehouse_props:
            WarehouseProps(self).build(self.belt_exit)
            # the reflective floor mirrors the belt legs as long thin streaks; matte looks right
            self.asset.find("./material[@name='floorplane']").set("reflectance", "0")

    def configure_location(self):
        self.floor.set("pos", array_to_string(self.bottom_pos))

        self.table_body.set("pos", array_to_string(self.center_pos))

        # Belt spans x in [x_min, half_x + belt_extension] in the body frame; x_min is
        # -half_x, or the start of the bend when there is an exit section.
        x_max = self.table_half_size[0] + self.belt_extension
        x_min = self.belt_exit.x_end if self.belt_exit is not None else -self.table_half_size[0]
        belt_half_x = (x_max - x_min) / 2.0
        belt_center_x = (x_max + x_min) / 2.0
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

        if self.belt_exit is not None:
            self._build_exit(self.belt_exit)

    # ---- bend, exit belt and crate --------------------------------------------------------
    BELT_FRICTION = "0.005 0.005 0.0001"   # near-frictionless; the environment drives bottles
    RAIL_FRICTION = "0.01 0.005 0.0001"

    def _add_box(self, name, pos, half, yaw=0.0, collide=True, material=None, rgba=None, friction=None):
        quat = array_to_string([np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)])
        common = dict(type="box", pos=array_to_string(pos), size=array_to_string(half), quat=quat)
        look = {"material": material} if material else {"rgba": array_to_string(rgba)}
        self.table_body.append(new_element(tag="geom", name=f"{name}_vis", group="1", contype="0",
                                           conaffinity="0", **common, **look))
        if collide:
            self.table_body.append(new_element(tag="geom", name=f"{name}_col", group="0", rgba="0 0 0 0",
                                               friction=friction or "0.8 0.005 0.0001", **common))

    def _build_exit(self, ex, n_seg=9):
        """Bend + exit belt + guide rails + crate, in the belt body's frame (top surface at +half_z)."""
        hz = self.table_half_size[2]
        top = hz                                   # belt surface height in the body frame
        floor_z = -self.table_offset[2] + hz       # world floor, in the body frame
        cx, cy = ex.center
        half_w = ex.width / 2
        dth = (np.pi / 2) / n_seg
        rail_h, rail_t = 0.02, 0.005               # rails rise 4 cm above the belt surface

        for i in range(n_seg):
            th = np.pi / 2 + (i + 0.5) * dth
            yaw = th + np.pi / 2                   # segment's long axis along the travel direction
            r_out = ex.radius + half_w
            p = [cx + ex.radius * np.cos(th), cy + ex.radius * np.sin(th), 0.0]
            half_len = r_out * np.tan(dth / 2) + 0.003
            self._add_box(f"belt_curve{i}", p, [half_len, half_w, hz], yaw, material="belt_mat",
                          friction=self.BELT_FRICTION)
            for side, r in (("in", ex.radius - half_w - rail_t), ("out", ex.radius + half_w + rail_t)):
                q = [cx + r * np.cos(th), cy + r * np.sin(th), top + rail_h]
                self._add_box(f"guide_curve_{side}{i}", q, [r * np.tan(dth / 2) + 0.002, rail_t, rail_h], yaw,
                              material="frame_mat", friction=self.RAIL_FRICTION)

        y_mid = (cy + ex.exit_y_end) / 2
        self._add_box("belt_exit", [ex.exit_x, y_mid, 0.0], [half_w, ex.length / 2, hz], material="belt_mat",
                      friction=self.BELT_FRICTION)
        for side in (-1, 1):
            self._add_box(f"guide_exit_{'in' if side < 0 else 'out'}",
                          [ex.exit_x + side * (half_w + rail_t), y_mid, top + rail_h],
                          [rail_t, ex.length / 2, rail_h], material="frame_mat", friction=self.RAIL_FRICTION)

        # Legs under the bend and the exit belt (visual only).
        leg_half = (-hz - floor_z) / 2
        leg_xy = [(cx + ex.radius * np.cos(t), cy + ex.radius * np.sin(t)) for t in (np.radians(135),)]
        leg_xy += [(ex.exit_x, y) for y in np.linspace(cy - 0.15, ex.exit_y_end + 0.1, 3)]
        for k, (x, y) in enumerate(leg_xy):
            for side in (-1, 1):
                self.table_body.append(new_element(
                    tag="geom", name=f"exit_leg{k}_{side + 1}", type="cylinder", group="1", contype="0",
                    conaffinity="0", material="frame_mat", size=array_to_string([0.018, leg_half]),
                    pos=array_to_string([x + side * 0.6 * half_w * (k > 0), y + side * 0.6 * half_w * (k == 0),
                                         floor_z + leg_half])))

        # Cardboard packing carton for good bottles, just past the end of the exit belt; its
        # rim sits below the belt surface so bottles slide off the belt and drop in. The flaps
        # are folded out and down (visual only); a wooden pallet and a steel stand carry it.
        for name, tex, attrs in (("crate_cardboard", "cardboard.png", dict(specular="0.05", shininess="0.05")),
                                 ("crate_wood", "pallet_wood.png", dict(specular="0.02", shininess="0.02")),
                                 ("crate_label", "shipping_label.png", dict(specular="0.1", shininess="0.1"))):
            self.asset.append(new_element(tag="texture", name=f"{name}_tex", type="2d",
                                          file=os.path.join(TEX_DIR, tex)))
            self.asset.append(new_element(tag="material", name=name, texture=f"{name}_tex", **attrs))
        c = ex.crate_half
        crate_y = ex.exit_y_end - c - 0.012
        floor_top = top - ex.crate_drop
        wall_h, t = 0.06, 0.005
        cx_, card = ex.exit_x, "crate_cardboard"
        self._add_box("crate_floor", [cx_, crate_y, floor_top - 0.01], [c + 0.01, c + 0.01, 0.01], material=card)
        walls = (
            ("crate_near", [cx_, crate_y + c + t, floor_top + wall_h], [c + 0.01, t, wall_h], (0, 1)),
            ("crate_far", [cx_, crate_y - c - t, floor_top + wall_h], [c + 0.01, t, wall_h], (0, -1)),
            ("crate_left", [cx_ - c - t, crate_y, floor_top + wall_h], [t, c + 0.01, wall_h], (-1, 0)),
            ("crate_right", [cx_ + c + t, crate_y, floor_top + wall_h], [t, c + 0.01, wall_h], (1, 0)),
        )
        rim = floor_top + 2 * wall_h
        flap_w = 0.11
        for name, pos, half, (ox, oy) in walls:
            self._add_box(name, pos, half, material=card)
            # the near flap hangs almost straight down so it clears the end of the belt
            a = np.radians(78 if name == "crate_near" else 28)
            hinge = np.array([pos[0] + ox * t, pos[1] + oy * t, rim])
            centre = hinge + np.array([ox, oy, 0.0]) * flap_w / 2 * np.cos(a) - [0, 0, flap_w / 2 * np.sin(a)]
            if oy:      # flap along x, rotate about x
                th = -oy * a
                quat, size = [np.cos(th / 2), np.sin(th / 2), 0, 0], [c + 0.012, flap_w / 2, 0.0025]
            else:       # flap along y, rotate about y
                ph = ox * a
                quat, size = [np.cos(ph / 2), 0, np.sin(ph / 2), 0], [flap_w / 2, c + 0.012, 0.0025]
            self.table_body.append(new_element(
                tag="geom", name=f"{name}_flap", type="box", group="1", contype="0", conaffinity="0",
                pos=array_to_string(centre), quat=array_to_string(quat), size=array_to_string(size),
                material=card))
        # packing tape down the outside of the right wall and a shipping label beside it
        self._add_box("crate_tape", [cx_ + c + 2 * t + 0.0005, crate_y + 0.07, floor_top + 0.045],
                      [0.0006, 0.022, 0.045], collide=False, rgba=[0.80, 0.64, 0.40, 1])
        self.table_body.append(new_element(
            tag="geom", name="crate_label_vis", type="box", group="1", contype="0", conaffinity="0",
            pos=array_to_string([cx_ + c + 2 * t + 0.0008, crate_y - 0.05, floor_top + 0.04]),
            size="0.055 0.034 0.0008", quat=array_to_string(facing_quat("+x")), material="crate_label"))
        # wooden pallet under the carton, on a steel stand
        pallet_top = floor_top - 0.02
        for dx in (-c + 0.02, 0.0, c - 0.02):
            self._add_box(f"crate_pallet_block{int(dx * 100)}", [cx_ + dx, crate_y, pallet_top - 0.033],
                          [0.03, c + 0.03, 0.022], collide=False, material="crate_wood")
        self._add_box("crate_pallet_deck", [cx_, crate_y, pallet_top - 0.006], [c + 0.04, c + 0.03, 0.006],
                      collide=False, material="crate_wood")
        stand_top = pallet_top - 0.055
        stand_half = (stand_top - floor_z) / 2
        self._add_box("crate_stand", [cx_, crate_y, floor_z + stand_half], [c * 0.8, c * 0.8, stand_half],
                      collide=False, material="frame_mat")

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
