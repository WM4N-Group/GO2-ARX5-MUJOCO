"""Regression checks for the measured box-task deployment contracts."""

import unittest
from unittest.mock import patch

import mujoco
import numpy as np
import torch

from check_box_push_policy import make_model
from check_box_support_sequence import surface_entry
from reconfigurable_navigation.box_climb_runtime import BoxClimbRuntime
from reconfigurable_navigation.box_navigation_runtime import BoxNavigationRuntime


class BoxRuntimeChecks(unittest.TestCase):
    def test_climb_entry_follows_surface_normal(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="box" pos="1 2 0.1" size="0.6 0.4 0.1" quat="0.7071067811865476 0 0 0.7071067811865476"/></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        entry, yaw = surface_entry(model, data, 0)
        np.testing.assert_allclose(entry, [1.0, 0.65], atol=1e-10)
        self.assertAlmostEqual(yaw, np.pi / 2.0)

    def setUp(self):
        torch.set_num_threads(1)
        self.model = make_model()
        self.data = mujoco.MjData(self.model)
        self.runtime = BoxClimbRuntime(model=self.model, data=self.data)
        self.runtime.joint_velocity_limits[12:] = 3.0
        self.data.qpos[2] = 0.33
        self.data.qpos[self.runtime.joint_qpos_adr] = self.runtime.default_qpos
        self.data.qpos[self.runtime.joint_qpos_adr[13:15]] = 0.15
        mujoco.mj_forward(self.model, self.data)
        self.runtime.activate()

    def test_first_delayed_command_uses_real_target_not_zero(self):
        target = self.data.qpos[self.runtime.joint_qpos_adr].copy() + 0.01
        step = mujoco.mj_step
        applied = []

        def inspect_torque(model, data):
            expected = self.runtime.kp * (target - data.qpos[self.runtime.joint_qpos_adr]) - self.runtime.kd * data.qvel[self.runtime.joint_dof_adr]
            expected = np.clip(expected, self.runtime.ctrl_low, self.runtime.ctrl_high)
            np.testing.assert_allclose(data.ctrl, expected, atol=1e-10)
            applied.append(data.ctrl.copy())
            step(model, data)

        with patch.object(mujoco, "mj_step", side_effect=inspect_torque):
            self.runtime._simulate_target(target)
        self.assertEqual(len(applied), self.runtime.decimation)

    def test_actuator_transfer_preserves_physics_and_owns_queue(self):
        self.runtime._simulate_target(self.runtime.default_qpos.copy())
        state_type = mujoco.mjtState.mjSTATE_INTEGRATION
        before = np.zeros(mujoco.mj_stateSize(self.model, state_type))
        mujoco.mj_getState(self.model, self.data, before, state_type)
        other = BoxClimbRuntime(model=self.model, data=self.data)
        other.inherit_actuator_state(self.runtime)
        after = np.zeros_like(before)
        mujoco.mj_getState(self.model, self.data, after, state_type)
        np.testing.assert_array_equal(before, after)
        np.testing.assert_array_equal(other.target_history[-1], self.runtime.target_history[-1])
        self.runtime.target_history[-1][0] += 1.0
        self.assertNotEqual(other.target_history[-1][0], self.runtime.target_history[-1][0])
        self.assertFalse(other._fresh_targets)
        isolated = BoxClimbRuntime(model=self.model, data=mujoco.MjData(self.model))
        with self.assertRaises(ValueError):
            isolated.inherit_actuator_state(self.runtime)

    def test_side_force_is_not_upward_foot_support(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><option gravity="0 0 0"/><worldbody><geom name="wall" type="box" pos="0.6 0 1" size="0.1 1 1"/><body pos="0.41 0 1"><freejoint/><geom name="foot" type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        runtime = object.__new__(BoxClimbRuntime)
        runtime.model, runtime.data = model, data
        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, 0, force)
        self.assertGreater(force[0], 2.0)
        self.assertEqual(runtime.support_contacts({1}, {0}), {})

    def test_upward_force_counts_as_foot_support(self):
        model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom name="floor" type="plane" size="1 1 0.1"/><body pos="0 0 0.09"><freejoint/><geom name="foot" type="sphere" size="0.1" mass="1"/></body></worldbody></mujoco>')
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        runtime = object.__new__(BoxClimbRuntime)
        runtime.model, runtime.data = model, data
        self.assertEqual(set(runtime.support_contacts({1}, {0})), {1})

    def test_nav_history_contains_blended_applied_commands(self):
        self.runtime._simulate_target(self.runtime.default_qpos.copy())
        navigation = BoxNavigationRuntime(self.runtime, blend_seconds=0.5)
        with torch.inference_mode():
            raw = navigation.policy(torch.from_numpy(navigation.observation()).unsqueeze(0))[0].numpy()
        applied = navigation.step(np.zeros(3))
        self.assertGreater(float(np.max(np.abs(applied - raw))), 1e-3)
        np.testing.assert_allclose(navigation.history[-1][42:60], applied, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()