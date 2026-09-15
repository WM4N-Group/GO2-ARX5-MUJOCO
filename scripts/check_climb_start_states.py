"""Check prepared-state mapping and support attribution without Isaac Sim."""

import ast
import hashlib
import json
from pathlib import Path
import runpy
import tempfile
from types import SimpleNamespace
import unittest

import torch
import yaml


MDP = Path(__file__).resolve().parents[1] / "source/LeggedManip_Lab/LeggedManip_Lab/tasks/manager_based/leggedmanip_lab/mdp"
STARTS = runpy.run_path(str(MDP / "climb_start_states.py"))
REWARDS = runpy.run_path(str(MDP / "box_rewards.py"))
definition = next(node for node in ast.parse((MDP / "box_climb.py").read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "supported_feet")
namespace = {"torch": torch, "SceneEntityCfg": object, "top_surface_support": REWARDS["top_surface_support"]}
exec(compile(ast.Module(body=[definition], type_ignores=[]), str(MDP / "box_climb.py"), "exec"), namespace)
supported_feet = namespace["supported_feet"]


class Scene(dict):
    pass


class Robot:
    def __init__(self, names):
        self.names = list(reversed(names))
        self.data = SimpleNamespace(default_joint_pos=torch.full((3, 18), 0.25))
        self.positions = torch.zeros(3, 18)
        self.velocities = torch.zeros(3, 18)
        self.poses = torch.zeros(3, 7)
        self.root_velocities = torch.zeros(3, 6)

    def find_joints(self, names, preserve_order):
        return [self.names.index(name) for name in names], names

    def write_root_pose_to_sim(self, pose, env_ids):
        self.poses[env_ids] = pose

    def write_root_velocity_to_sim(self, velocity, env_ids):
        self.root_velocities[env_ids] = velocity

    def write_joint_state_to_sim(self, positions, velocities, env_ids):
        self.positions[env_ids] = positions
        self.velocities[env_ids] = velocities


class PreparedStartChecks(unittest.TestCase):
    def test_export_binds_training_and_evaluation_state_files(self):
        self.write()
        params = Path(self.directory.name) / "params"
        params.mkdir()
        (params / "env.yaml").write_text(yaml.safe_dump({"prepared_start_path": str(self.path), "prepared_start_phase": "ground"}))
        physics = {"prepared_start_path": str(self.path), "prepared_start_phase": "ground", "prepared_start_sha256": hashlib.sha256(self.path.read_bytes()).hexdigest()}
        collect = runpy.run_path(str(Path(__file__).with_name("export_box_climb_actor.py")))["prepared_start_sources"]
        sources = collect(Path(self.directory.name) / "model_0.pt", [{"actual_physics": physics}])
        self.assertEqual([source["role"] for source in sources], ["training", "evaluation"])
        self.assertEqual(sources[0]["file"], sources[1]["file"])
        physics["prepared_start_sha256"] = "changed"
        with self.assertRaises(ValueError):
            collect(Path(self.directory.name) / "model_0.pt", [{"actual_physics": physics}])

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / "starts.json"
        self.state = {
            "phase": "ground", "joint_names": [f"test_joint{index}" for index in range(18)],
            "base_pose": [0.04, -0.01, 0.29, 1.0, 0.0, 0.0, 0.0],
            "base_velocity_world": [0.01, 0.0, 0.0, 0.0, 0.0, 0.02],
            "joint_position": [index / 100 for index in range(18)],
            "joint_velocity": [index / 1000 for index in range(18)],
        }

    def write(self, succeeded=True):
        self.path.write_text(json.dumps({"results": [{"stages": [{"succeeded": succeeded, "prepared_state": self.state}]}]}))

    def test_reset_maps_named_joints_without_changing_defaults_or_other_envs(self):
        self.write()
        robot = Robot(self.state["joint_names"])
        scene = Scene(robot=robot)
        scene.env_origins = torch.tensor([[0.0, 0.0, 0.0], [4.0, 2.0, 0.20], [8.0, 2.0, 0.40]])
        env = SimpleNamespace(cfg=SimpleNamespace(prepared_start_path=str(self.path), prepared_start_phase="ground"), device="cpu", scene=scene)
        STARTS["reset_prepared_start"](env, torch.tensor([1, 2]))
        torch.testing.assert_close(robot.positions[1], torch.tensor(self.state["joint_position"]).flip(0))
        torch.testing.assert_close(robot.velocities[2], torch.tensor(self.state["joint_velocity"]).flip(0))
        torch.testing.assert_close(robot.poses[1, :3], torch.tensor([4.04, 1.99, 0.49]))
        torch.testing.assert_close(robot.root_velocities[2], torch.tensor(self.state["base_velocity_world"]))
        torch.testing.assert_close(robot.positions[0], torch.zeros(18))
        torch.testing.assert_close(robot.poses[0], torch.zeros(7))
        torch.testing.assert_close(robot.data.default_joint_pos, torch.full((3, 18), 0.25))

    def test_disabled_profile_does_not_access_physics(self):
        STARTS["reset_prepared_start"](SimpleNamespace(cfg=SimpleNamespace(prepared_start_path="")), torch.tensor([0]))

    def test_missing_phase_or_failed_preparation_is_rejected(self):
        self.write()
        with self.assertRaises(ValueError):
            STARTS["load_start_states"](self.path, "platform")
        self.write(succeeded=False)
        with self.assertRaises(ValueError):
            STARTS["load_start_states"](self.path, "ground")

    def test_nonfinite_or_unnormalized_state_is_rejected(self):
        for field, index, value in (("base_pose", 3, 2.0), ("joint_position", 4, float("nan"))):
            original = self.state[field][index]
            self.state[field][index] = value
            self.write()
            with self.assertRaises(ValueError):
                STARTS["load_start_states"](self.path, "ground")
            self.state[field][index] = original

    def test_following_platform_is_not_the_target_support_height(self):
        origin = torch.tensor([[10.0, 20.0, 0.2]])
        feet = torch.tensor([[[1.1, -0.15, 0.2225], [1.1, 0.15, 0.2225], [1.6, -0.15, 0.2225], [1.6, 0.15, 0.2225]]]) + origin[:, None]
        hits = torch.tensor([[[1.2, 0.0, 0.20], [2.4, 0.0, 0.40]]]) + origin[:, None]
        robot = SimpleNamespace(data=SimpleNamespace(body_pos_w=feet))
        sensor = SimpleNamespace(data=SimpleNamespace(net_forces_w=torch.tensor([[[0.0, 0.0, 10.0]]]).expand(1, 4, 3)))
        scanner = SimpleNamespace(data=SimpleNamespace(ray_hits_w=hits))
        scene = Scene(robot=robot, contact_forces=sensor, height_scanner=scanner)
        scene.sensors = {"contact_forces": sensor}
        scene.env_origins = origin
        config = SimpleNamespace(approach_distance=0.75, box_length=1.2, box_width=1.2)
        env = SimpleNamespace(scene=scene, cfg=SimpleNamespace(scene=SimpleNamespace(terrain=SimpleNamespace(terrain_generator=SimpleNamespace(sub_terrains={"box": config})))))
        asset = SimpleNamespace(name="robot", body_ids=[0, 1, 2, 3])
        contact = SimpleNamespace(name="contact_forces", body_ids=[0, 1, 2, 3])
        torch.testing.assert_close(supported_feet(env, asset, contact), torch.ones(1, 4, dtype=torch.bool))
        scanner.data.ray_hits_w = hits[:, 1:]
        torch.testing.assert_close(supported_feet(env, asset, contact), torch.zeros(1, 4, dtype=torch.bool))


if __name__ == "__main__":
    unittest.main()