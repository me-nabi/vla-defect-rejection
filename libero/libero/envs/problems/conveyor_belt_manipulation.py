import numpy as np
from robosuite.utils.mjcf_utils import new_site
from libero.libero.envs.bddl_base_domain import BDDLBaseDomain, register_problem
from libero.libero.envs.robots import *
from libero.libero.envs.objects import *
from libero.libero.envs.predicates import *
from libero.libero.envs.regions import *
from libero.libero.envs.utils import rectangle2xyrange


@register_problem
class Conveyor_Belt_Manipulation(BDDLBaseDomain):
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
        self._belt_geom_id = model.geom_name2id("table_collision")

    # Bottles tilted more than this are treated as knocked over and not carried.
    _MAX_CARRY_TILT_COS = np.cos(np.radians(30))

    def _bottles_on_belt(self):
        """(qpos, qvel) addresses of bottles currently in contact with the belt surface."""
        model, data = self.sim.model, self.sim.data
        on_belt = set()
        for i in range(data.ncon):
            contact = data.contact[i]
            if contact.geom1 == self._belt_geom_id:
                other = contact.geom2
            elif contact.geom2 == self._belt_geom_id:
                other = contact.geom1
            else:
                continue
            root = model.body_rootid[model.geom_bodyid[other]]
            if root in self._bottle_root_to_adr:
                on_belt.add(self._bottle_root_to_adr[root])
        return on_belt

    def _pre_action(self, action, policy_step=False):
        super()._pre_action(action, policy_step=policy_step)
        if not self.belt_active:
            return
        # Conveyor transport. The belt geom is static, so friction against it would
        # tip a bottle that is pushed along at belt speed. Instead, a bottle that is
        # touching the belt and roughly upright is carried at the belt velocity
        # (no sideways drift, no spin) with its tilt removed (yaw kept). Bottles
        # lifted off the belt (e.g. grasped) or knocked over are left to the physics.
        qpos_all, qvel_all = self.sim.data.qpos, self.sim.data.qvel
        for qpos_adr, qvel_adr in self._bottles_on_belt():
            w, x, y, z = qpos_all[qpos_adr + 3 : qpos_adr + 7]  # MuJoCo quat: w, x, y, z
            up_z = 1.0 - 2.0 * (x * x + y * y)  # z-component of the bottle's up axis
            if up_z < self._MAX_CARRY_TILT_COS:
                continue
            yaw_norm = np.hypot(w, z)
            qpos_all[qpos_adr + 3 : qpos_adr + 7] = [w / yaw_norm, 0.0, 0.0, z / yaw_norm]

            qvel = qvel_all[qvel_adr : qvel_adr + 6].copy()
            qvel[0] = -self.belt_speed  # along the belt (-x)
            qvel[1] = 0.0               # across the belt
            qvel[3:6] = 0.0             # no rotation
            qvel_all[qvel_adr : qvel_adr + 6] = qvel

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
