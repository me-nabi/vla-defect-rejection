"""Debug script to understand gripper behavior and bottle geometry."""
import sys
sys.path.insert(0, "/home/nabi/LIBERO")

import numpy as np
from libero.libero.envs import TASK_MAPPING

BDDL_FILE = "/home/nabi/LIBERO/libero/libero/bddl_files/defect_rejection/pick_defective_bottle_from_belt.bddl"

CONTROLLER_CONFIG = {
    "type": "OSC_POSE",
    "input_max": 1, "input_min": -1,
    "output_max": [0.12, 0.12, 0.12, 0.5, 0.5, 0.5],
    "output_min": [-0.12, -0.12, -0.12, -0.5, -0.5, -0.5],
    "kp": 180, "damping": 1,
    "impedance_mode": "fixed",
    "kp_limits": [0, 300], "damping_limits": [0, 10],
    "position_limits": None, "orientation_limits": None,
    "uncouple_pos_ori": True,
    "control_delta": True,
    "interpolation": None,
    "ramp_ratio": 0.2,
}

env = TASK_MAPPING["conveyor_belt_manipulation"](
    bddl_file_name=BDDL_FILE,
    robots=["Panda"],
    controller_configs=CONTROLLER_CONFIG,
    has_renderer=False,
    has_offscreen_renderer=True,
    render_camera="agentview",
    ignore_done=True,
    use_camera_obs=True,
    camera_heights=256,
    camera_widths=256,
    camera_names=["agentview", "robot0_eye_in_hand"],
    control_freq=20,
    belt_active=False,
    belt_speed=0.0,
)

obs = env.reset()

bottle_pos = obs["glass_bottle_1_pos"]
eef_pos = obs["robot0_eef_pos"]
print(f"Bottle pos: {bottle_pos}")
print(f"EEF pos: {eef_pos}")
print(f"Gripper qpos: {obs['robot0_gripper_qpos']}")
print(f"Gripper qvel: {obs['robot0_gripper_qvel']}")

# Check all observation keys
print(f"\nObs keys: {sorted(obs.keys())}")

# Get bottle joint info
for name in env.sim.model.joint_names:
    if 'bottle' in name or 'gripper' in name:
        jid = env.sim.model.joint_name2id(name)
        print(f"Joint: {name}, id={jid}, dofadr={env.sim.model.jnt_dofadr[jid]}")

# Get bottle body info
for name in env.sim.model.body_names:
    if 'bottle' in name:
        bid = env.sim.model.body_name2id(name)
        bpos = env.sim.data.body_xpos[bid]
        print(f"Body: {name}, id={bid}, pos={bpos}")

# Get geom info for bottle
for i, name in enumerate(env.sim.model.geom_names):
    if name and ('bottle' in name.lower() or 'glass' in name.lower()):
        gpos = env.sim.model.geom_pos[i]
        gsize = env.sim.model.geom_size[i]
        print(f"Geom: {name}, pos={gpos}, size={gsize}")

print("\n--- Testing gripper action ---")
# Move to bottle position and try grasping
def compute_action(current, target, gain=10.0):
    delta = target - current
    return np.clip(delta * gain, -1.0, 1.0)

# Phase 1: Move above bottle
print("\nMoving above bottle...")
for i in range(40):
    eef_pos = obs["robot0_eef_pos"]
    bottle_pos = obs["glass_bottle_1_pos"]
    target = np.array([bottle_pos[0], bottle_pos[1], bottle_pos[2] + 0.12])
    action = np.zeros(7)
    action[:3] = compute_action(eef_pos, target, gain=12.0)
    action[6] = -1.0  # open gripper
    obs, _, _, _ = env.step(action)
    if i % 10 == 0:
        print(f"  Step {i}: eef={obs['robot0_eef_pos']}, gripper_qpos={obs['robot0_gripper_qpos']}")

# Phase 2: Descend to bottle
print("\nDescending to bottle...")
for i in range(40):
    eef_pos = obs["robot0_eef_pos"]
    bottle_pos = obs["glass_bottle_1_pos"]
    target = np.array([bottle_pos[0], bottle_pos[1], bottle_pos[2] + 0.045])
    action = np.zeros(7)
    action[:3] = compute_action(eef_pos, target, gain=12.0)
    action[6] = -1.0  # keep open
    obs, _, _, _ = env.step(action)
    if i % 10 == 0:
        print(f"  Step {i}: eef={obs['robot0_eef_pos']}, gripper_qpos={obs['robot0_gripper_qpos']}, bottle={obs['glass_bottle_1_pos']}")

# Phase 3: Close gripper
print("\nClosing gripper on bottle...")
for i in range(40):
    eef_pos = obs["robot0_eef_pos"]
    bottle_pos = obs["glass_bottle_1_pos"]
    action = np.zeros(7)
    action[6] = 1.0  # close gripper
    obs, _, _, _ = env.step(action)
    if i % 10 == 0:
        print(f"  Step {i}: eef={obs['robot0_eef_pos']}, gripper_qpos={obs['robot0_gripper_qpos']}, bottle={obs['glass_bottle_1_pos']}")

# Phase 4: Lift
print("\nLifting...")
for i in range(40):
    eef_pos = obs["robot0_eef_pos"]
    bottle_pos = obs["glass_bottle_1_pos"]
    target = np.array([eef_pos[0], eef_pos[1], 1.15])
    action = np.zeros(7)
    action[:3] = compute_action(eef_pos, target, gain=10.0)
    action[6] = 1.0  # keep gripper closed
    obs, _, _, _ = env.step(action)
    if i % 10 == 0:
        print(f"  Step {i}: eef={obs['robot0_eef_pos']}, gripper_qpos={obs['robot0_gripper_qpos']}, bottle={obs['glass_bottle_1_pos']}")

print(f"\nFinal check: bottle_z={obs['glass_bottle_1_pos'][2]:.4f} (surface was 0.915)")
if obs['glass_bottle_1_pos'][2] > 0.93:
    print("SUCCESS: Bottle was lifted!")
else:
    print("FAIL: Bottle was NOT lifted")

env.close()
