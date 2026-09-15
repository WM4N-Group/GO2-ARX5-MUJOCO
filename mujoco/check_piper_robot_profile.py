"""Check PIPER profile mass, forward kinematics, Jacobians and gravity."""

import json
import unittest

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

from reconfigurable_navigation.piper_locomotion_runtime import PiperLocomotionRuntime
from reconfigurable_navigation.piper_robot_profile import PROFILE_PATH


def profile_frames(profile, joint_angles):
    frames = {"base_link": np.eye(4)}
    frames["base_link"][2, 3] = 1.0
    joint_by_body = {}
    for entry in profile["joints"]:
        name = entry["name"]
        body = name.removesuffix("_joint") if "_joint" in name else name.replace("joint", "link")
        joint_by_body[body] = entry
    pending = {body["name"]: body for body in profile["bodies"] if body["name"] != "base_link"}
    while pending:
        resolved = []
        for name, entry in pending.items():
            if entry["parent"] not in frames:
                continue
            local = np.eye(4)
            local[:3, :3] = Rotation.from_quat(entry["quaternion"], scalar_first=True).as_matrix()
            local[:3, 3] = entry["position"]
            if name in joint_by_body:
                joint = joint_by_body[name]
                local[:3, :3] = local[:3, :3] @ Rotation.from_rotvec(np.array(joint["axis"]) * joint_angles[joint["name"]]).as_matrix()
            frames[name] = frames[entry["parent"]] @ local
            resolved.append(name)
        if not resolved:
            raise ValueError("Disconnected profile graph")
        for name in resolved:
            del pending[name]
    return frames


class PiperProfileChecks(unittest.TestCase):
    def test_profile_matches_compiled_model_and_virtual_work(self):
        runtime = PiperLocomotionRuntime(profile_path=PROFILE_PATH)
        profile = json.loads(PROFILE_PATH.read_text())
        self.assertAlmostEqual(float(runtime.model.body_mass.sum()), profile["robot_mass"], places=7)
        self.assertEqual(runtime.model.nu, 18)
        generator = np.random.default_rng(12)
        for _sample in range(12):
            angles = {joint["name"]: generator.uniform(*joint["range"]) for joint in profile["joints"]}
            runtime.data.qpos[:7] = [0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0]
            for name, value in angles.items():
                runtime.data.joint(name).qpos[0] = value
            runtime.data.qvel.fill(0.0)
            mujoco.mj_forward(runtime.model, runtime.data)
            frames = profile_frames(profile, angles)
            for name, transform in frames.items():
                np.testing.assert_allclose(runtime.data.body(name).xpos, transform[:3, 3], atol=1e-7)
                np.testing.assert_allclose(runtime.data.body(name).xmat.reshape(3, 3), transform[:3, :3], atol=1e-7)
            jacobian = np.zeros((3, runtime.model.nv))
            hand_id = runtime.model.body("end_effector").id
            mujoco.mj_jacBody(runtime.model, runtime.data, jacobian, None, hand_id)
            for name in ("joint2", "joint3", "FL_thigh_joint"):
                samples = []
                for direction in (-1.0, 1.0):
                    perturbed = angles.copy()
                    perturbed[name] += direction * 1e-6
                    shifted = profile_frames(profile, perturbed)
                    energy = sum(body["mass"] * 9.81 * (shifted[body["name"]] @ np.array([*body["center_of_mass"], 1.0]))[2] for body in profile["bodies"])
                    samples.append((shifted["end_effector"][:3, 3], energy))
                dof = int(runtime.model.joint(name).dofadr[0])
                np.testing.assert_allclose(jacobian[:, dof], (samples[1][0] - samples[0][0]) / 2e-6, atol=1e-6)
                self.assertAlmostEqual(float(runtime.data.qfrc_bias[dof]), (samples[1][1] - samples[0][1]) / 2e-6, places=5)


if __name__ == "__main__":
    unittest.main()