"""Scripted pick-and-place of one bottle into the reject bin.

Shared by pipeline_demo.py and inference_smolvla.py, so both use the same,
collision-aware motion. The target comes from the vision system (inspection/vision.py):
where the overhead camera located the bottle and how tall it is. The picker never reads the
simulator's bottle positions; it checks a grasp by how far the fingers closed.

    approach -> descend -> grasp -> lift -> transfer -> lower -> settle -> release -> retreat

The carried bottle travels high enough that its base clears the bin rim and the bottles
standing on the belt or in the bin (at least SAFE_Z; higher for tall bottles). It only goes
down vertically, once it is over its release slot in the bin. Successive bottles use
different slots so they don't land on top of each other. A failed grasp (the fingers closed
on nothing) is retried once.

Gripper convention: -1 = open, +1 = close. Actions are OSC_POSE deltas in [-1, 1].
Every action also levels the gripper so it points straight down. With zero
rotation commands the OSC just keeps whatever orientation it was knocked into;
a tilted gripper swings the bottle and needs more reach.
"""
import numpy as np
import robosuite.utils.transform_utils as T

# Heights in world metres. The belt surface is at z = 0.90 (cell calibration); bottles are
# 9-16 cm tall depending on the design. The bin rim is at ~0.99.
BELT_Z = 0.90
SAFE_Z = 1.16       # minimum carry height of the gripper
CARRY_CLEARANCE_Z = 1.08   # the carried bottle's base stays above this: bin rim (~0.99) and the tops of
                           # bottles standing in the bin or on the belt nearby
APPROACH_DZ = 0.15  # hover at least this far above the belt (and 5 cm above the bottle top) before descending
GRASP_FRACTION = 0.69   # grasp at this fraction of the bottle height: the shoulder / upper body. The palm
                        # may meet the cap first; the descent then stops there (stall check).
MIN_GRASP_DZ = 0.05     # never aim the gripper lower than this above the belt. The depth camera doesn't see
                        # clear glass/PET or the drink (like real RGB-D sensors), only the cap and label, so
                        # a bottle without its cap can measure as short as its label; the palm then stops
                        # on the glass lip during the descent and the grasp closes there.
RELEASE_GAP = 0.016 # release with the bottle's base this far above the bin base, so it lands upright
GRIP_HOLDING = 0.006    # finger opening (m) above which the gripper is holding something

# Pick a defect once it reaches this stretch of belt (world x). The arm is mounted at
# x = -0.66, and further out than x ~ 0.08 a low grasp straightens the elbow to its joint
# limit, so the arm can no longer move freely. Bottles flow toward the robot, so just wait.
PICK_ZONE_X = (-0.04, 0.04)

# Release spots relative to the bin centre, along its 15 cm long (x) axis.
# Two bottles (4.4 cm wide) fit side by side.
BIN_SLOTS = [(-0.035, 0.0), (0.035, 0.0)]

GRASP_STEPS = 30
SETTLE_STEPS = 10   # hold still above the slot before opening, so a swinging bottle doesn't tip
CARRY_VMAX = 0.5    # slower moves while holding a bottle: it hangs from a neck grasp and swings
LOWER_VMAX = 0.3
RELEASE_STEPS = 20
MAX_ATTEMPTS = 2

STATUS = {
    "approach": "POSITIONING OVER TARGET",
    "descend": "DESCENDING TO TARGET",
    "grasp": "GRASPING DEFECTIVE BOTTLE",
    "lift": "REMOVING DEFECT FROM LINE",
    "transfer": "MOVING TO REJECT BIN",
    "lower": "LOWERING INTO REJECT BIN",
    "settle": "LOWERING INTO REJECT BIN",
    "release": "DEPOSITING IN REJECT BIN",
    "retreat": "RETURNING TO STANDBY",
    "done": "RETURNING TO STANDBY",
}


def p_action(current, target, gain, vmax=1.0):
    """Proportional position action, clipped to +-vmax (the controller accepts [-1, 1])."""
    return np.clip((np.asarray(target) - np.asarray(current)) * gain, -vmax, vmax)


def level_action(eef_quat, gain=1.5):
    """Rotation action that turns the gripper's approach (z) axis back to straight down.

    OSC_POSE takes world-frame axis-angle deltas, with input 1.0 = 0.5 rad. Yaw is left free.
    """
    z_axis = T.quat2mat(eef_quat)[:, 2]
    down = np.array([0.0, 0.0, -1.0])
    axis = np.cross(z_axis, down)
    sin_a, cos_a = np.linalg.norm(axis), float(np.dot(z_axis, down))
    if sin_a < 1e-6:
        return np.zeros(3)
    angle = np.arctan2(sin_a, cos_a)
    return np.clip(gain * axis / sin_a * angle, -1.0, 1.0)


class ScriptedPicker:
    """Step-by-step controller for one pick. Call step(obs) each env step until done.

    target: what the vision system measured, dict(label, xy, top_z).
    bin_xy, bin_z: where the reject box stands (cell calibration)."""

    def __init__(self, target, bin_xy, bin_z, slot=0, log=print):
        self.target = target["label"]
        self.xy = np.asarray(target["xy"], float)
        self.top_z = float(target["top_z"])
        height = self.top_z - BELT_Z
        self.grasp_z = BELT_Z + max(GRASP_FRACTION * height, MIN_GRASP_DZ)
        self.approach_z = max(BELT_Z + APPROACH_DZ, self.top_z + 0.05)
        self.bin_xy, self.bin_z = np.asarray(bin_xy, float), float(bin_z)
        self.slot_offset = np.array(BIN_SLOTS[slot % len(BIN_SLOTS)])
        self.log = log
        self._last_z = None
        self._still_steps = 0
        self._prev_eef = None
        self.drop_xy = None
        self.phase = "approach"
        self.steps_in_phase = 0
        self.attempt = 1
        self.hold_dz = self.grasp_z - BELT_Z     # gripper height above the bottle base while carrying
        self.carry_z = SAFE_Z
        self.succeeded = False

    @property
    def done(self):
        return self.phase == "done"

    @property
    def status(self):
        return STATUS[self.phase]

    def _goto(self, phase, note=""):
        self.log(f"    {self.phase} -> {phase}{(' ' + note) if note else ''}")
        self.phase = phase
        self.steps_in_phase = 0
        self._still_steps = 0

    def _stalled(self, eef_z):
        """True once the gripper has stopped moving vertically for 5 steps (blocked by contact)."""
        if self._last_z is not None and abs(eef_z - self._last_z) < 0.001:
            self._still_steps += 1
        else:
            self._still_steps = 0
        self._last_z = eef_z
        return self.steps_in_phase > 5 and self._still_steps >= 5

    @staticmethod
    def finger_opening(obs):
        q = obs["robot0_gripper_qpos"]
        return float(abs(q[0]) + abs(q[1]))

    def step(self, obs):
        eef = obs["robot0_eef_pos"]
        speed = 0.0 if self._prev_eef is None else float(np.linalg.norm(eef - self._prev_eef))
        self._prev_eef = np.array(eef)
        action = np.zeros(7)
        action[6] = -1.0
        t = self.steps_in_phase
        self.steps_in_phase += 1
        x, y = self.xy

        if self.phase == "approach":
            target = [x, y, self.approach_z]
            action[:3] = p_action(eef, target, gain=12.0)
            close_xy = np.linalg.norm(eef[:2] - self.xy) < 0.015
            if (close_xy and abs(eef[2] - target[2]) < 0.02) or t > 100:
                self._goto("descend", "(timeout)" if t > 100 else "")

        elif self.phase == "descend":
            target = [x, y, self.grasp_z]
            action[:3] = p_action(eef, target, gain=10.0)
            stalled = self._stalled(eef[2])
            if eef[2] < target[2] + 0.01 or stalled or t > 60:
                self.hold_dz = float(eef[2] - BELT_Z)
                self._goto("grasp", f"(gripper {self.hold_dz:.3f} above the belt{', stalled' if stalled else ''})")

        elif self.phase == "grasp":
            target = [x, y, BELT_Z + self.hold_dz]
            action[:3] = p_action(eef, target, gain=4.0)
            action[6] = 1.0
            if t >= GRASP_STEPS:
                self._goto("lift")

        elif self.phase == "lift":
            self.carry_z = max(SAFE_Z, CARRY_CLEARANCE_Z + self.hold_dz)
            action[:3] = p_action(eef, [eef[0], eef[1], self.carry_z], gain=10.0)
            action[6] = 1.0
            if eef[2] > self.carry_z - 0.02 or t > 80:
                opening = self.finger_opening(obs)
                if opening < GRIP_HOLDING:
                    if self.attempt < MAX_ATTEMPTS:
                        self.attempt += 1
                        self._goto("approach", f"(fingers closed on nothing ({opening:.3f}), retrying)")
                    else:
                        self._goto("retreat", "(grasp missed, giving up)")
                else:
                    self.drop_xy = self.bin_xy + self.slot_offset
                    self._goto("transfer", f"(holding, finger opening {opening:.3f})")

        elif self.phase == "transfer":
            action[:3] = p_action(eef, [self.drop_xy[0], self.drop_xy[1], self.carry_z], gain=10.0, vmax=CARRY_VMAX)
            action[6] = 1.0
            arrived = np.linalg.norm(eef[:2] - self.drop_xy) < 0.01 and speed < 0.002
            if arrived or t > 120:
                self._goto("lower", "(timeout)" if t > 120 else "")

        elif self.phase == "lower":
            target = [self.drop_xy[0], self.drop_xy[1], self.bin_z + self.hold_dz + RELEASE_GAP]
            action[:3] = p_action(eef, target, gain=8.0, vmax=LOWER_VMAX)
            action[6] = 1.0
            stalled = self._stalled(eef[2])
            if eef[2] < target[2] + 0.005 or stalled or t > 60:
                self._goto("settle", "(stalled)" if stalled else "")

        elif self.phase == "settle":
            target = [self.drop_xy[0], self.drop_xy[1], self.bin_z + self.hold_dz + RELEASE_GAP]
            action[:3] = p_action(eef, target, gain=4.0, vmax=LOWER_VMAX)
            action[6] = 1.0
            if t >= SETTLE_STEPS:
                self._goto("release")

        elif self.phase == "release":
            action[6] = -1.0
            if t >= RELEASE_STEPS:
                self.succeeded = True
                self._goto("retreat")

        elif self.phase == "retreat":
            action[:3] = p_action(eef, [eef[0], eef[1], max(SAFE_Z, self.carry_z - 0.04)], gain=8.0)
            if eef[2] > SAFE_Z - 0.02 or t > 60:
                self._goto("done")

        action[3:6] = level_action(obs["robot0_eef_quat"])
        return action
