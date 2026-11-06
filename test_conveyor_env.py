import numpy as np
from libero.libero.envs import TASK_MAPPING

print("Registered problem types:", list(TASK_MAPPING.keys()))
assert "conveyor_belt_manipulation" in TASK_MAPPING, "Conveyor_Belt_Manipulation not registered!"
print("Conveyor_Belt_Manipulation registered OK")

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

print("Environment created successfully!")

obs = env.reset()
print(f"Observation keys: {list(obs.keys())}")
print(f"agentview_image shape: {obs['agentview_image'].shape}")
print(f"robot0_eye_in_hand_image shape: {obs['robot0_eye_in_hand_image'].shape}")
print(f"robot0_eef_pos: {obs['robot0_eef_pos']}")

action = np.zeros(7)
for i in range(5):
    obs, reward, done, info = env.step(action)

print(f"5 steps done. Belt active: bottles should be moving.")

env.close()
print("Test PASSED - conveyor belt environment works!")
