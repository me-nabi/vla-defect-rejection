"""Debug the scripted policy with corrected positions."""
import numpy as np
from PIL import Image
from libero.libero.envs import TASK_MAPPING

BDDL_FILE = "/home/nabi/LIBERO/libero/libero/bddl_files/defect_rejection/pick_defective_bottle_from_belt.bddl"

env = TASK_MAPPING["conveyor_belt_manipulation"](
    bddl_file_name=BDDL_FILE,
    robots=["Panda"],
    controller_configs={"type": "OSC_POSE", "input_max": 1, "input_min": -1,
                        "output_max": [0.05, 0.05, 0.05, 0.5, 0.5, 0.5],
                        "output_min": [-0.05, -0.05, -0.05, -0.5, -0.5, -0.5],
                        "kp": 150, "damping": 1, "impedance_mode": "fixed",
                        "kp_limits": [0, 300], "damping_limits": [0, 10],
                        "position_limits": None, "orientation_limits": None,
                        "uncouple_pos_ori": True, "control_delta": True,
                        "interpolation": None, "ramp_ratio": 0.2},
    has_renderer=False, has_offscreen_renderer=True, render_camera="agentview",
    ignore_done=True, use_camera_obs=True,
    camera_heights=256, camera_widths=256,
    camera_names=["agentview", "robot0_eye_in_hand"],
    reward_shaping=True, control_freq=20,
    belt_active=False,
)

obs = env.reset()
bottle_pos = obs["glass_bottle_1_pos"].copy()
bin_pos = obs["reject_bin_1_pos"].copy()
print(f"Bottle: {bottle_pos}")
print(f"Bin:    {bin_pos}")
print(f"EEF:    {obs['robot0_eef_pos']}")

def move_to(env, obs, target, gripper, label, max_steps=120, gain=12.0):
    for i in range(max_steps):
        eef = obs["robot0_eef_pos"]
        delta = target - eef
        action = np.zeros(7)
        action[:3] = np.clip(delta * gain, -1, 1)
        action[6] = gripper
        obs, _, _, _ = env.step(action)
        dist = np.linalg.norm(eef - target)
        if i % 20 == 0:
            bottle = obs["glass_bottle_1_pos"]
            print(f"  [{label}] step {i}: eef={np.round(eef,3)}, dist={dist:.4f}, bottle_z={bottle[2]:.4f}")
        if dist < 0.008:
            print(f"  [{label}] reached in {i} steps, dist={dist:.4f}")
            break
    return obs

def do_gripper(env, obs, cmd, label, steps=25):
    for _ in range(steps):
        action = np.zeros(7)
        action[6] = cmd
        obs, _, _, _ = env.step(action)
    gripper = obs["robot0_gripper_qpos"]
    bottle = obs["glass_bottle_1_pos"]
    print(f"  [{label}] gripper_qpos={np.round(gripper,4)}, bottle={np.round(bottle,4)}")
    return obs

# Grasp center of the bottle body (main cylinder center at z_offset=0.035)
grasp_z = bottle_pos[2] + 0.05

# Phase 1: Move above bottle (gripper open = -1)
target_above = np.array([bottle_pos[0], bottle_pos[1], grasp_z + 0.12])
print(f"\n--- Phase 1: Move above bottle (target={np.round(target_above,3)}) ---")
obs = move_to(env, obs, target_above, -1.0, "above")

# Phase 2: Descend to grasp (gripper open = -1)
target_grasp = np.array([bottle_pos[0], bottle_pos[1], grasp_z])
print(f"\n--- Phase 2: Descend (target={np.round(target_grasp,3)}) ---")
obs = move_to(env, obs, target_grasp, -1.0, "descend")
img = obs["agentview_image"][::-1]
Image.fromarray(img).save("/home/nabi/LIBERO/debug_pre_grasp.png")

# Phase 3: Close gripper (+1 = close)
print("\n--- Phase 3: Close gripper ---")
obs = do_gripper(env, obs, 1.0, "close", steps=30)
img = obs["agentview_image"][::-1]
Image.fromarray(img).save("/home/nabi/LIBERO/debug_post_grasp.png")

# Phase 4: Lift
target_lift = obs["robot0_eef_pos"].copy()
target_lift[2] = bottle_pos[2] + 0.22
print(f"\n--- Phase 4: Lift (target_z={target_lift[2]:.3f}) ---")
obs = move_to(env, obs, target_lift, 1.0, "lift")
bottle_after = obs["glass_bottle_1_pos"]
lifted = bottle_after[2] > bottle_pos[2] + 0.03
print(f"  Bottle z: {bottle_pos[2]:.4f} → {bottle_after[2]:.4f}  LIFTED={lifted}")

if not lifted:
    print("\n  ** GRASP FAILED — aborting **")
    env.close()
    exit()

# Phase 5: Move above bin
target_bin_above = np.array([bin_pos[0], bin_pos[1], bin_pos[2] + 0.20])
print(f"\n--- Phase 5: Move above bin (target={np.round(target_bin_above,3)}) ---")
obs = move_to(env, obs, target_bin_above, 1.0, "to_bin")
img = obs["agentview_image"][::-1]
Image.fromarray(img).save("/home/nabi/LIBERO/debug_above_bin.png")

# Phase 6: Lower into bin
target_drop = np.array([bin_pos[0], bin_pos[1], bin_pos[2] + 0.10])
print(f"\n--- Phase 6: Lower into bin ---")
obs = move_to(env, obs, target_drop, 1.0, "lower")

# Phase 7: Release
print("\n--- Phase 7: Release ---")
obs = do_gripper(env, obs, -1.0, "release", steps=30)

# Settle
for _ in range(30):
    obs, _, _, _ = env.step(np.zeros(7))

bottle_final = obs["glass_bottle_1_pos"]
bin_final = obs["reject_bin_1_pos"]
success = env._check_success()
print(f"\nFinal bottle: {np.round(bottle_final, 4)}")
print(f"Final bin:    {np.round(bin_final, 4)}")
print(f"Distance:     {np.linalg.norm(bottle_final - bin_final):.4f}")
print(f"SUCCESS:      {success}")

img = obs["agentview_image"][::-1]
Image.fromarray(img).save("/home/nabi/LIBERO/debug_final.png")
env.close()
