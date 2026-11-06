import numpy as np
from PIL import Image
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
    camera_heights=512,
    camera_widths=512,
    camera_names=["agentview", "robot0_eye_in_hand"],
    reward_shaping=True,
    control_freq=20,
)

obs = env.reset()

img = obs["agentview_image"]
img = img[::-1]
Image.fromarray(img).save("/home/nabi/LIBERO/conveyor_agentview_initial.png")
print("Saved initial agentview")

hand_img = obs["robot0_eye_in_hand_image"]
hand_img = hand_img[::-1]
Image.fromarray(hand_img).save("/home/nabi/LIBERO/conveyor_hand_initial.png")
print("Saved initial hand view")

action = np.zeros(7)
for i in range(50):
    obs, _, _, _ = env.step(action)

img_after = obs["agentview_image"][::-1]
Image.fromarray(img_after).save("/home/nabi/LIBERO/conveyor_agentview_after50steps.png")
print("Saved after 50 steps (bottles should have moved on belt)")

bottle_pos = obs["glass_bottle_1_pos"]
print(f"Bottle position after 50 steps: {bottle_pos}")

env.close()
print("Done!")
