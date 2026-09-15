"""Existing NAV actor on the box-task physical actuator profile."""

from types import SimpleNamespace

from .locomotion_runtime import LocomotionRuntime, MJ_POLICY_INDICES, POLICY_PATH


class BoxNavigationRuntime(LocomotionRuntime):
    def __init__(self, physics, policy_path=POLICY_PATH, *, blend_seconds=0.0):
        if blend_seconds < 0.0:
            raise ValueError("Blend duration must be nonnegative")
        self.physics = physics
        self.blend_seconds = blend_seconds
        joint_id = int(physics.model.body_jntadr[physics.base_id])
        env = SimpleNamespace(
            model=physics.model, data=physics.data, robot_joint_id=joint_id,
            robot_qpos_adr=int(physics.model.jnt_qposadr[joint_id]),
            deploy_cfg={
                "simulation_dt": physics.sim_dt, "control_decimation": physics.decimation,
                "default_angles": physics.default_qpos, "kps": physics.kp, "kds": physics.kd,
                "action_scale": physics.action_scale,
            },
        )
        super().__init__(env, policy_path)
        self._start_target = physics.target_history[-1].copy()
        self._last_target = self._start_target.copy()
        self._activation_elapsed = 0.0

    def _simulate_target(self, target_mj):
        self._activation_elapsed += self.control_dt
        fraction = min(self._activation_elapsed / self.blend_seconds, 1.0) if self.blend_seconds else 1.0
        self._last_target = self._start_target + fraction * (target_mj - self._start_target)
        self.physics._simulate_target(self._last_target)

    def step(self, velocity_command, ee_command=None):
        super().step(velocity_command, ee_command)
        self.last_action = ((self._last_target - self.default_qpos) / self.action_scale)[MJ_POLICY_INDICES]
        self.history[-1] = self._single_observation()
        return self.last_action.copy()