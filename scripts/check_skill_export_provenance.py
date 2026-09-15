"""Check that PIPER exports reject changed assets and robot identities."""

from copy import deepcopy
import hashlib
from pathlib import Path
import tempfile
import unittest

import yaml

from skill_export_provenance import piper_robot_provenance


class ExportProvenanceChecks(unittest.TestCase):
    def test_matching_profiles_pass_and_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / "assets/go2_piper/robot.usd"
            asset.parent.mkdir(parents=True)
            asset.write_bytes(b"test asset content")
            params = root / "run/params"
            params.mkdir(parents=True)
            params.joinpath("env.yaml").write_text(yaml.safe_dump({"scene": {"robot": {"spawn": {"usd_path": str(asset)}}}}))
            physics = {"robot_asset_sha256": {str(asset.relative_to(root)): hashlib.sha256(asset.read_bytes()).hexdigest()},
                       "robot_joint_names": [f"joint_{index}" for index in range(18)], "robot_effort_limits": [5.0]*18,
                       "robot_joint_defaults": [0.0]*18}
            report = {"actual_physics": physics}
            checkpoint = root / "run/model.pt"
            self.assertEqual(piper_robot_provenance(root, checkpoint, [report, deepcopy(report)])["robot"], "go2_piper")
            changed = deepcopy(report)
            changed["actual_physics"]["robot_effort_limits"][0] = 100.0
            with self.assertRaisesRegex(ValueError, "profiles differ"):
                piper_robot_provenance(root, checkpoint, [report, changed])
            asset.write_bytes(b"changed asset")
            with self.assertRaisesRegex(ValueError, "asset differs"):
                piper_robot_provenance(root, checkpoint, [report])
            params.joinpath("env.yaml").write_text(yaml.safe_dump({"scene": {"robot": {"spawn": {"usd_path": "/assets/go2_arx5/robot.usd"}}}}))
            with self.assertRaisesRegex(ValueError, "does not use PIPER"):
                piper_robot_provenance(root, checkpoint, [report])


if __name__ == "__main__":
    unittest.main()