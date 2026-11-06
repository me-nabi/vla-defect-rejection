"""Quick debug script to understand object positions and env dynamics."""
import numpy as np
from libero.libero.envs import TASK_MAPPING

bddl_file = "/home/nabi/LIBERO/libero/libero/bddl_files/defect_rejection/pick_defective_bottle_from_belt.bddl"

env = TASK_MAPPING["conveyor_belt_manipulation"](
    bddl_file_name=bddl_file,
    robots=["Panda"],
    controller_configs={"type": "OSC_POSE", "input_max": 1, "input_min": -1,
                        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                        "kp": 150, "damping": 1,
                        "impedance_mode": "fixed",
                        "kp_limits": [0, 300],
                        "damping_limits": [0, 10],
                        "position_limits": None,
                        "orientation_limits": None,
                        "uncouple_pos_ori": True,
                        "control_delta": True,
                        "interpolation": None,
                        "ramp_ratio": 0.2},
    has_renderer=False,
    has_offscreen_renderer=True,
    render_camera="agentview",
    ignore_done=True,
    use_camera_obs=True,
    camera_heights=256,
    camera_widths=256,
    camera_names=["agentview", "robot0_eye_in_hand"],
    reward_shaping=True,
    control_freq=20,
)

for ep in range(3):
    obs = env.reset()
    print(f"\n=== Episode {ep} ===")
    print(f"  EEF pos:      {obs['robot0_eef_pos']}")
    print(f"  EEF quat:     {obs['robot0_eef_quat']}")
    print(f"  Gripper qpos: {obs['robot0_gripper_qpos']}")
    print(f"  Bottle pos:   {obs['glass_bottle_1_pos']}")
    print(f"  Bottle quat:  {obs['glass_bottle_1_quat']}")
    print(f"  Bin pos:      {obs['reject_bin_1_pos']}")
    print(f"  Bin quat:     {obs['reject_bin_1_quat']}")
    print(f"  Action dim:   {env.action_dim}")
    print(f"  Action spec:  low={env.action_spec[0]}, high={env.action_spec[1]}")

env.close()
