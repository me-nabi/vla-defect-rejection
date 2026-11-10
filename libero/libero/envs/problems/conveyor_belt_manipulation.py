import numpy as np
import robosuite.utils.transform_utils as T
from robosuite.utils.mjcf_utils import new_element, new_site
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange
from libero.libero.envs.inspection_defects import DEFECT_TYPES, BottleAppearance
from libero.libero.envs.arenas.conveyor_belt_arena import BeltExit


def _look_at_quat(pos, target):
    """MuJoCo camera quaternion (w, x, y, z) for a camera at `pos` looking at `target`, z up."""
    z_cam = np.asarray(pos, float) - np.asarray(target, float)
    z_cam /= np.linalg.norm(z_cam)
    x_cam = np.cross([0.0, 0.0, 1.0], z_cam)
    x_cam /= np.linalg.norm(x_cam)
    y_cam = np.cross(z_cam, x_cam)
    return T.convert_quat(T.mat2quat(np.stack([x_cam, y_cam, z_cam], axis=1)), to="wxyz")


@register_problem
class Conveyor_Belt_Manipulation(BDDLBaseDomain):
    # Inspection station: two cameras facing each other across the belt at this x.
    # Bottles are photographed as they pass it, before reaching the robot.
    INSPECTION_X = 0.20
    INSPECTION_CAMERAS = ("inspect_front", "inspect_back")
    INSPECTION_CAM_DIST = 0.30      # camera distance from the belt centre line (m)
    INSPECTION_CAM_FOVY = 34        # frames the tallest design (16 cm) with a margin
    INSPECTION_CAM_Z = 1.0          # camera height; they aim at the middle of the frame:
    INSPECTION_TARGET_Z = 0.985     # belt surface 0.90 + 8.5 cm
    # Overhead RGB-D camera for the bottle detector: looks straight down at the belt from
    # just upstream of the inspection station to past the robot's pick zone.
    # Close-up cameras at the pick area: same optics as the inspection station, centred on the
    # stretch where an arriving bottle is judged (x = 0.02-0.10, just before it stops at ~0.04),
    # so a policy without memory can judge the bottle it is about to pick. No extra light (it
    # would change every other view) and no housing geometry.
    PICK_CAM_X = 0.06
    PICK_CAMERAS = ("pick_front", "pick_back")
    TRACK_CAMERA = "track_cam"
    TRACK_CAM_POS = (0.20, 0.0, 1.55)
    TRACK_CAM_FOVY = 42             # vertical (across the belt); 4:3 images span x = -0.19 .. 0.59
    TRACK_CAM_SIZE = (384, 512)     # (height, width)

    def __init__(self, bddl_file_name, *args, **kwargs):
        self.workspace_name = "conveyor_belt"
        self.visualization_sites_list = []
        self.table_full_size = (1.0, 0.4, 0.05)
        self.table_offset = (0, 0, 0.90)
        self.z_offset = 0.01 - self.table_full_size[2]

        self.belt_speed = kwargs.pop("belt_speed", 0.02)
        self.belt_active = kwargs.pop("belt_active", True)
        # Extra belt length (m) on the upstream (+x) end; 0 keeps the original 1 m belt.
        belt_extension = kwargs.pop("belt_extension", 0.0)
        # Bend + exit belt + crate after the robot (instead of the belt running into the
        # robot's pedestal). The bend and exit belt are separate conveyors: they keep running
        # while the main belt is paused for a pick, at `exit_speed` (default 3x belt_speed).
        self.belt_layout = BeltExit() if kwargs.pop("belt_exit", True) else None
        self.exit_speed = kwargs.pop("exit_speed", None)
        # Inspection bottles: probabilities of 0, 1, 2, ... defective bottles per episode,
        # and which defect types to draw from.
        self.defect_count_probs = kwargs.pop("defect_count_probs", (0.15, 0.40, 0.45))
        self.defect_types = tuple(kwargs.pop("defect_types", DEFECT_TYPES))
        # Bottle designs on the line and how often each appears (see bottle_shapes.py).
        variants = kwargs.pop("bottle_variants", None)
        if variants is not None:
            InspectionBottle.VARIANT_WEIGHTS = dict(variants) if isinstance(variants, dict) \
                else {v: 1.0 for v in variants}
        self.bottle_defects = {}       # inspection bottle name -> defect type, or None if good
        self._bottle_looks = {}

        kwargs.update(
            {"robots": [f"Mounted{robot_name}" for robot_name in kwargs["robots"]]}
        )
        kwargs.update({"workspace_offset": self.table_offset})
        kwargs.update({"arena_type": "conveyor"})

        if "scene_xml" not in kwargs or kwargs["scene_xml"] is None:
            kwargs.update({"scene_xml": "scenes/libero_conveyor_belt.xml"})
        if "scene_properties" not in kwargs or kwargs["scene_properties"] is None:
            kwargs.update(
                {
                    "scene_properties": {
                        "floor_style": "dark",
                        "wall_style": "dark-gray-plaster",
                    }
                }
            )
        kwargs["scene_properties"] = {
            **kwargs["scene_properties"],
            "belt_extension": belt_extension,
            "belt_exit": self.belt_layout,
        }

        self._bottle_joint_ids = []

        super().__init__(bddl_file_name, *args, **kwargs)

    def _setup_references(self):
        super()._setup_references()
        model = self.sim.model
        self._bottle_joint_ids = []
        # root body id -> (qpos address, qvel address) of that bottle's free joint
        self._bottle_root_to_adr = {}
        for obj_name in self.objects_dict:
            if "bottle" in obj_name:
                joint_name = f"{obj_name}_joint0"
                try:
                    joint_id = model.joint_name2id(joint_name)
                except ValueError:
                    continue
                self._bottle_joint_ids.append(joint_id)
                root = model.body_rootid[model.jnt_bodyid[joint_id]]
                self._bottle_root_to_adr[root] = (
                    model.jnt_qposadr[joint_id],
                    model.jnt_dofadr[joint_id],
                )
        # Conveyor surfaces: the main belt plus the bend and exit segments.
        belt_names = ["table_collision"] + [
            model.geom_id2name(g) for g in range(model.ngeom)
            if (model.geom_id2name(g) or "").startswith(("belt_curve", "belt_exit")) and model.geom_id2name(g).endswith("_col")
        ]
        self._belt_geom_ids = {model.geom_name2id(n) for n in belt_names}
        # The belt geoms are static, so real friction against them would brake a moving
        # bottle at its base and tip it over. Belt surfaces and bottles get near-zero
        # friction instead, and _pre_action drives each bottle like belt friction would.
        # Contacts use the larger of the two frictions, so grasps (finger pads: 2), the bin
        # and the crate still grip bottles normally.
        for g in self._belt_geom_ids:
            model.geom_friction[g, 0] = self.BELT_SURFACE_FRICTION
        self._bottle_geoms = [g for g in range(model.ngeom)
                              if model.body_rootid[model.geom_bodyid[g]] in self._bottle_root_to_adr
                              and model.geom_contype[g] != 0]
        for g in self._bottle_geoms:
            model.geom_friction[g, 0] = self.BOTTLE_FRICTION
        # The free joint sits on a massless outer body whose origin is the bottle's base;
        # the drive force must act on the body that carries the mass (at its centre of mass),
        # or it would tip the bottle.
        self._mass_body = {}
        for root in self._bottle_root_to_adr:
            bodies = [b for b in range(model.nbody) if model.body_rootid[b] == root]
            self._mass_body[root] = max(bodies, key=lambda b: model.body_mass[b])
        self._driven = set()
        self._bottle_looks = {
            name: BottleAppearance(self.sim, name, obj.shape)
            for name, obj in self.objects_dict.items()
            if obj.category_name == "inspection_bottle"
        }

    def _reset_internal(self):
        super()._reset_internal()
        if self._bottle_looks:
            self.randomize_bottles()

    # ---- inspection bottles -------------------------------------------------------------
    def randomize_bottles(self, rng=None):
        """Choose which inspection bottles are defective (and how) and redraw all of them.

        Uses the global numpy RNG by default, so seeding np.random before reset() makes the
        whole episode reproducible.
        """
        rng = np.random.default_rng(np.random.randint(2**31)) if rng is None else rng
        names = sorted(self._bottle_looks)
        probs = np.asarray(self.defect_count_probs[: len(names) + 1], float)
        n_bad = int(rng.choice(len(probs), p=probs / probs.sum()))
        bad = set(rng.choice(names, size=n_bad, replace=False)) if n_bad else set()
        for name in names:
            options = [d for d in self.defect_types if d in self.bottle_shapes[name].defects]
            self.set_bottle_defect(name, rng.choice(options) if name in bad and options else None, rng)
            # Random yaw, so labels and defects face any direction (bottles stay upright).
            joint = self.objects_dict[name].joints[-1]
            qpos = self.sim.data.get_joint_qpos(joint).copy()
            yaw = rng.uniform(0, 2 * np.pi)
            qpos[3:7] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
            self.sim.data.set_joint_qpos(joint, qpos)
        self.sim.forward()

    def set_bottle_defect(self, name, defect, rng=None):
        """Redraw one inspection bottle as good (defect=None) or with the given defect type."""
        defect = None if defect is None else str(defect)
        self._bottle_looks[name].apply(self.sim, defect, rng)
        self.bottle_defects[name] = defect
        self.sim.forward()

    @property
    def inspection_bottles(self):
        return sorted(self._bottle_looks)

    @property
    def bottle_shapes(self):
        """Inspection bottle name -> its design (BottleShape)."""
        return {name: look.shape for name, look in self._bottle_looks.items()}

    @property
    def defective_bottles(self):
        return sorted(n for n, d in self.bottle_defects.items() if d is not None)

    # ---- conveyor physics ---------------------------------------------------------------
    # A bottle touching a belt surface is pulled towards the local belt velocity by a force
    # at its centre of mass, capped at what friction could transmit (TRACTION_MU * m * g).
    # Blocked bottles therefore slip instead of being shoved over, fallen bottles ride the
    # belt like upright ones, and a stopped belt holds bottles in place.
    BELT_SURFACE_FRICTION = 0.005
    BOTTLE_FRICTION = 0.01
    TRACTION_GAIN = 40.0     # 1/s: how quickly a bottle takes up the belt speed
    TRACTION_MU = 0.8
    CENTERING_GAIN = 3.0     # 1/s: pulls bottles onto the centreline of the bend and exit belt
    FUNNEL_LENGTH = 0.2      # the main belt starts centring bottles this far before the bend

    def _contacts_on_belt(self):
        """Root body ids of bottles currently touching any conveyor surface."""
        model, data = self.sim.model, self.sim.data
        roots = set()
        for i in range(data.ncon):
            c = data.contact[i]
            if c.geom1 in self._belt_geom_ids:
                other = c.geom2
            elif c.geom2 in self._belt_geom_ids:
                other = c.geom1
            else:
                continue
            root = model.body_rootid[model.geom_bodyid[other]]
            if root in self._bottle_root_to_adr:
                roots.add(root)
        return roots

    def belt_velocity(self, x, y):
        """Conveyor surface velocity (vx, vy) at world position (x, y)."""
        main = self.belt_speed if self.belt_active else 0.0
        lay = self.belt_layout
        if lay is None or x >= lay.x_end:
            vy = 0.0
            if lay is not None:     # funnel bottles towards the bend's centreline
                vy = -self.CENTERING_GAIN * y * np.clip((lay.x_end + self.FUNNEL_LENGTH - x) / self.FUNNEL_LENGTH, 0, 1)
            return np.array([-main, vy])
        speed = self.exit_speed if self.exit_speed is not None else 3.0 * self.belt_speed
        cx, cy = lay.center
        if y >= cy:                 # in the bend
            dx, dy = x - cx, y - cy
            r = np.hypot(dx, dy)
            tangent = np.array([-dy, dx]) / max(r, 1e-6)
            normal = np.array([dx, dy]) / max(r, 1e-6)
            return speed * tangent - self.CENTERING_GAIN * (r - lay.radius) * normal
        return np.array([-self.CENTERING_GAIN * (x - lay.exit_x), -speed])   # exit belt

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)
        model, data = self.sim.model, self.sim.data
        on_belt = self._contacts_on_belt()
        for root in self._driven - on_belt:
            data.xfrc_applied[self._mass_body[root]] = 0.0
        g = -model.opt.gravity[2]
        for root in on_belt:
            _, dof = self._bottle_root_to_adr[root]
            mass = model.body_subtreemass[root]
            body = self._mass_body[root]
            x, y = data.xipos[body][:2]
            dv = self.belt_velocity(x, y) - data.qvel[dof:dof + 2]
            force = mass * self.TRACTION_GAIN * dv
            cap = self.TRACTION_MU * mass * g
            norm = np.linalg.norm(force)
            if norm > cap:
                force *= cap / norm
            data.xfrc_applied[body, :2] = force
            # damp spinning about the vertical, as belt friction would (free-joint angular
            # velocity is in the body frame; the applied torque is in the world frame)
            w_world = data.xmat[root].reshape(3, 3) @ data.qvel[dof + 3:dof + 6]
            data.xfrc_applied[body, 5] = -self.TRACTION_GAIN * 1e-4 * w_world[2]
        self._driven = on_belt

    def _load_fixtures_in_arena(self, mujoco_arena):
        for fixture_category in list(self.parsed_problem["fixtures"].keys()):
            if fixture_category == "table":
                continue
            for fixture_instance in self.parsed_problem["fixtures"][fixture_category]:
                self.fixtures_dict[fixture_instance] = get_object_fn(fixture_category)(
                    name=fixture_instance,
                    joints=None,
                )

    def _load_objects_in_arena(self, mujoco_arena):
        objects_dict = self.parsed_problem["objects"]
        for category_name in objects_dict.keys():
            for object_name in objects_dict[category_name]:
                self.objects_dict[object_name] = get_object_fn(category_name)(
                    name=object_name
                )

    def _load_sites_in_arena(self, mujoco_arena):
        object_sites_dict = {}
        region_dict = self.parsed_problem["regions"]
        for object_region_name in list(region_dict.keys()):
            if "conveyor_belt" in object_region_name:
                ranges = region_dict[object_region_name]["ranges"][0]
                assert ranges[2] >= ranges[0] and ranges[3] >= ranges[1]
                zone_size = ((ranges[2] - ranges[0]) / 2, (ranges[3] - ranges[1]) / 2)
                zone_centroid_xy = (
                    (ranges[2] + ranges[0]) / 2,
                    (ranges[3] + ranges[1]) / 2,
                )
                target_zone = TargetZone(
                    name=object_region_name,
                    rgba=region_dict[object_region_name]["rgba"],
                    zone_size=zone_size,
                    zone_centroid_xy=zone_centroid_xy,
                )
                object_sites_dict[object_region_name] = target_zone

                mujoco_arena.table_body.append(
                    new_site(
                        name=target_zone.name,
                        pos=target_zone.pos,
                        quat=target_zone.quat,
                        rgba=target_zone.rgba,
                        size=target_zone.size,
                        type="box",
                    )
                )
                continue
            for query_dict in [self.objects_dict, self.fixtures_dict]:
                for (name, body) in query_dict.items():
                    try:
                        if "worldbody" not in list(body.__dict__.keys()):
                            continue
                    except:
                        continue
                    for part in body.worldbody.find("body").findall(".//body"):
                        sites = part.findall(".//site")
                        joints = part.findall("./joint")
                        if sites == []:
                            break
                        for site in sites:
                            site_name = site.get("name")
                            if site_name == object_region_name:
                                site_kwargs = dict(
                                    name=site_name,
                                    parent_name=body.name,
                                    joints=[joint.get("name") for joint in joints],
                                    object_properties=body.object_properties,
                                )
                                if site.get("size") is not None:
                                    site_kwargs["size"] = site.get("size")
                                if site.get("rgba") is not None:
                                    site_kwargs["rgba"] = site.get("rgba")
                                if site.get("type") is not None:
                                    site_kwargs["site_type"] = site.get("type")
                                if site.get("pos") is not None:
                                    site_kwargs["site_pos"] = site.get("pos")
                                if site.get("quat") is not None:
                                    site_kwargs["site_quat"] = site.get("quat")
                                object_sites_dict[object_region_name] = SiteObject(
                                    **site_kwargs
                                )
        self.object_sites_dict = object_sites_dict

        for query_dict in [self.fixtures_dict, self.objects_dict]:
            for name, body in query_dict.items():
                if body.object_properties["vis_site_names"] != {}:
                    self.visualization_sites_list.append(name)

    def _check_success(self):
        if self._bottle_looks:
            # Inspection line: every defective bottle rejected, and no good bottle.
            region = "reject_bin_1_contain_region"
            in_bin = {n: bool(self._eval_predicate(["in", n, region])) for n in self._bottle_looks}
            return all(in_bin[n] == (self.bottle_defects.get(n) is not None) for n in in_bin)
        goal_state = self.parsed_problem["goal_state"]
        result = True
        for state in goal_state:
            result = self._eval_predicate(state) and result
        return result

    def _eval_predicate(self, state):
        if len(state) == 3:
            predicate_fn_name = state[0]
            object_1_name = state[1]
            object_2_name = state[2]
            return eval_predicate_fn(
                predicate_fn_name,
                self.object_states_dict[object_1_name],
                self.object_states_dict[object_2_name],
            )
        elif len(state) == 2:
            predicate_fn_name = state[0]
            object_name = state[1]
            return eval_predicate_fn(
                predicate_fn_name, self.object_states_dict[object_name]
            )

    def _post_process(self):
        super()._post_process()
        self.set_visualization()

    def set_visualization(self):
        for object_name in self.visualization_sites_list:
            for _, (site_name, site_visible) in (
                self.get_object(object_name).object_properties["vis_site_names"].items()
            ):
                vis_g_id = self.sim.model.site_name2id(site_name)
                if ((self.sim.model.site_rgba[vis_g_id][3] <= 0) and site_visible) or (
                    (self.sim.model.site_rgba[vis_g_id][3] > 0) and not site_visible
                ):
                    self.sim.model.site_rgba[vis_g_id][3] = (
                        1 - self.sim.model.site_rgba[vis_g_id][3]
                    )

    def _setup_camera(self, mujoco_arena):
        mujoco_arena.set_camera(
            camera_name="agentview",
            pos=[0.45, 0.0, 1.45],
            quat=[
                0.6380177736282349,
                0.3048497438430786,
                0.30484986305236816,
                0.6380177736282349,
            ],
        )
        mujoco_arena.set_camera(
            camera_name="frontview",
            pos=[0.8, 0.0, 1.35],
            quat=[0.56, 0.43, 0.43, 0.56],
        )
        # Close side view: inspection station, robot pick and reject bin.
        pos, target = [0.15, 1.3, 1.62], [-0.25, 0.0, 1.08]
        mujoco_arena.set_camera(camera_name="sideview", pos=pos, quat=_look_at_quat(pos, target),
                                camera_attribs={"fovy": "47"})
        # Overview of the whole line: inspection station, robot and bin, bend, exit belt, carton.
        pos, target = [0.8, -2.0, 2.0], [-0.25, -0.8, 0.78]
        mujoco_arena.set_camera(camera_name="lineview", pos=pos, quat=_look_at_quat(pos, target),
                                camera_attribs={"fovy": "40"})
        # Inspection station: two cameras facing each other across the belt, slightly above
        # bottle mid-height, each with its own light, so every side of a bottle is seen.
        x, d = self.INSPECTION_X, self.INSPECTION_CAM_DIST
        target = [x, 0.0, self.INSPECTION_TARGET_Z]
        for name, side in zip(self.INSPECTION_CAMERAS, (-1.0, 1.0)):
            pos = [x, side * d, self.INSPECTION_CAM_Z]
            mujoco_arena.set_camera(
                camera_name=name, pos=pos, quat=_look_at_quat(pos, target),
                camera_attribs={"fovy": str(self.INSPECTION_CAM_FOVY)},
            )
            mujoco_arena.worldbody.append(new_element(
                tag="light", name=f"{name}_light", pos=f"{x - 0.05} {side * 0.25} 1.15",
                dir=f"0.15 {-side * 0.85} -0.5", directional="false", castshadow="false",
                diffuse="0.45 0.45 0.45", specular="0.3 0.3 0.3",
            ))
        x = self.PICK_CAM_X
        target = [x, 0.0, self.INSPECTION_TARGET_Z]
        for name, side in zip(self.PICK_CAMERAS, (-1.0, 1.0)):
            pos = [x, side * d, self.INSPECTION_CAM_Z]
            mujoco_arena.set_camera(camera_name=name, pos=pos, quat=_look_at_quat(pos, target),
                                    camera_attribs={"fovy": str(self.INSPECTION_CAM_FOVY)})
        # Overhead RGB-D camera for the bottle detector, under the inspection gantry's light bar.
        # Identity orientation = looking straight down, image right = +x (downstream is left).
        cx, cy, cz = self.TRACK_CAM_POS
        mujoco_arena.set_camera(camera_name=self.TRACK_CAMERA, pos=list(self.TRACK_CAM_POS), quat=[1, 0, 0, 0],
                                camera_attribs={"fovy": str(self.TRACK_CAM_FOVY)})
        for k, (pos, size, rgba) in enumerate((
                ([cx, cy, cz + 0.026], [0.035, 0.03, 0.02], "0.15 0.15 0.16 1"),        # body, under the light bar
                ([cx, cy, cz + 0.004], [0.012, 0.012, 0.004], "0.05 0.05 0.06 1"))):    # lens
            mujoco_arena.worldbody.append(new_element(
                tag="geom", name=f"track_cam_housing{k}", type="box", pos=" ".join(map(str, pos)),
                size=" ".join(map(str, size)), rgba=rgba, group="1", contype="0", conaffinity="0"))
