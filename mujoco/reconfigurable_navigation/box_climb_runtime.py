"""CLIMB runtime with the box-task actuator startup contract."""

import mujoco
import numpy as np

from .climb_runtime import ClimbRuntime


class BoxClimbRuntime(ClimbRuntime):
    def support_contacts(self, feet, surfaces):
        forces = {foot: np.zeros(3) for foot in feet}
        heights = {foot: [] for foot in feet}
        wrench = np.zeros(6)
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if not pair.intersection(surfaces):
                continue
            for foot in pair.intersection(feet):
                mujoco.mj_contactForce(self.model, self.data, index, wrench)
                normal = contact.frame.reshape(3, 3)[0] * wrench[0]
                if foot == contact.geom1:
                    normal = -normal
                forces[foot] += normal
                if normal[2] > 0.0:
                    heights[foot].append(float(contact.pos[2]))
        return {foot: min(heights[foot]) for foot in feet if forces[foot][2] > 2.0 and heights[foot]}

    def inherit_actuator_state(self, source):
        if self.model is not source.model or self.data is not source.data:
            raise ValueError("Actuator continuation requires the same physical world")
        self.target_history.clear()
        self.target_history.extend(target.copy() for target in source.target_history)
        self._fresh_targets = source._fresh_targets

    def activate(self):
        super().activate()
        self._fresh_targets = True

    def _simulate_target(self, target):
        if self._fresh_targets:
            self.target_history.clear()
            self.target_history.extend(target.copy() for _ in range(self.target_history.maxlen))
            self._fresh_targets = False
        super()._simulate_target(target)
        if getattr(self, "on_step", None) is not None:
            self.on_step(self)