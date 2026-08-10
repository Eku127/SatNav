#!/usr/bin/env python3
"""Fast regression tests for public trajectory-generation behavior."""

import json
import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from omegaconf import OmegaConf
from PIL import Image

from applications.trajectory_generation import generate_parallel
from applications.trajectory_generation.generate import build_parser
from applications.trajectory_generation.runner import SatNavTrajectoryRunner
from applications.trajectory_generation.utils import (
    PUBLIC_ANNOTATION_FIELDS,
    annotation_artifacts_complete,
    build_annotation,
    build_summary_entry,
    episode_generation_digest,
    episode_source_digest,
    format_episode_dirname,
    generation_contract_digest,
    public_annotation,
    strict_json_loads,
)
from satnav.navigation import ReferencePathFollower
from satnav.task.actions import Action, INITIAL_ACTION_INDEX, encode_action
from satnav.task.config import get_episode_success_distance

from tests.test_path_follower import FakeSimulator


EARTH_RADIUS_METERS = 6371000.0


def point_north(distance_meters):
    angular_distance = distance_meters / EARTH_RADIUS_METERS
    return [0.0, math.degrees(angular_distance), 0.0]


def make_annotation(scene_id="Rome-1"):
    episode_dirname = format_episode_dirname(scene_id, "satnav", 7)
    return build_annotation(
        episode_idx=7,
        trajectory_id="trajectory-7",
        episode_dirname=episode_dirname,
        instruction_text="stop here",
        actions=[INITIAL_ACTION_INDEX, encode_action(Action.STOP)],
    )


def create_frames(output_path: Path, annotation):
    rgb_dir = output_path / annotation["video"] / "rgb"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    for frame_index in range(1, annotation["steps"] + 2):
        Image.new("RGB", (2, 2), color=(frame_index, 0, 0)).save(
            rgb_dir / f"{frame_index:03d}.jpg", format="JPEG"
        )
    return rgb_dir


def make_episode(scene_id="Rome-1", instruction="stop here"):
    return SimpleNamespace(
        scene_id=scene_id,
        episode_id="episode-7",
        trajectory_id="trajectory-7",
        instruction=SimpleNamespace(instruction_text=instruction),
        reference_path=[[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        goals=[SimpleNamespace(position=[1.0, 1.0, 0.0])],
        start_position=[0.0, 0.0, 0.0],
        start_rotation=0.0,
        trajectory_type="Road",
    )


def make_generation_config(root: Path):
    scenes = root / "scenes"
    scenes.mkdir(exist_ok=True)
    (scenes / "Rome-1.tif").write_bytes(b"scene-content")
    return OmegaConf.create(
        {
            "ENVIRONMENT": {"MAX_EPISODE_STEPS": 10},
            "SIMULATOR": {
                "TYPE": "SatSim-v0",
                "FORWARD_STEP_SIZE": 10,
                "TURN_ANGLE": 15,
                "RGB_SENSOR": {"WIDTH": 4, "HEIGHT": 4, "HFOV": 90},
            },
            "TASK": {
                "SUCCESS_DISTANCE": {"DEFAULT": 10, "Road": 10},
                "MEASUREMENTS": ["DISTANCE_TO_GOAL", "SUCCESS", "SPL"],
            },
            "DATASET": {"SCENES_DIR": str(scenes)},
        }
    )


class LandmarkSuccessPriorityTests(unittest.TestCase):
    def test_type_specific_success_distance_uses_canonical_config_value(self):
        config = OmegaConf.create(
            {
                "TASK": {
                    "SUCCESS_DISTANCE": {
                        "DEFAULT": 10.0,
                        "LandmarkSet": 3.0,
                    }
                }
            }
        )

        self.assertEqual(
            get_episode_success_distance(config, "LandmarkSet"),
            3.0,
        )
        self.assertEqual(
            get_episode_success_distance(config, "unknown"),
            10.0,
        )

    def test_cli_omission_preserves_config_value(self):
        args = build_parser().parse_args(
            ["--config", "task.yaml", "--output_dir", "output"]
        )
        self.assertIsNone(args.landmark_success)

        runner = SatNavTrajectoryRunner.__new__(SatNavTrajectoryRunner)
        runner.config = OmegaConf.create(
            {"TASK": {"SUCCESS_DISTANCE": {"DEFAULT": 10.0, "LandmarkSet": 3.0}}}
        )
        runner._apply_success_distance_overrides(args.landmark_success)

        self.assertEqual(runner.config.TASK.SUCCESS_DISTANCE.LandmarkSet, 3.0)

    def test_explicit_cli_value_overrides_config(self):
        args = build_parser().parse_args(
            [
                "--config",
                "task.yaml",
                "--output_dir",
                "output",
                "--landmark_success",
                "2",
            ]
        )
        runner = SatNavTrajectoryRunner.__new__(SatNavTrajectoryRunner)
        runner.config = OmegaConf.create(
            {"TASK": {"SUCCESS_DISTANCE": {"DEFAULT": 10.0, "LandmarkSet": 3.0}}}
        )
        runner._apply_success_distance_overrides(args.landmark_success)

        self.assertEqual(runner.config.TASK.SUCCESS_DISTANCE.LandmarkSet, 2.0)


class AnnotationSchemaTests(unittest.TestCase):
    def test_public_annotation_and_summary_do_not_leak_scene_path(self):
        private_scene = "/mnt/private/satnav/scenes/Rome-1"
        annotation = make_annotation(private_scene)
        summary = build_summary_entry(
            annotation,
            scene_id=private_scene,
            episode_id="episode-7",
        )

        self.assertEqual(tuple(annotation), PUBLIC_ANNOTATION_FIELDS)
        self.assertFalse(Path(annotation["video"]).is_absolute())
        self.assertEqual(summary["scene_id"], "Rome-1")
        self.assertNotIn("/mnt/private", json.dumps(annotation))
        self.assertNotIn("/mnt/private", json.dumps(summary))

    def test_canonical_action_codec_is_unchanged(self):
        self.assertEqual(
            [
                encode_action(Action.STOP),
                encode_action(Action.MOVE_FORWARD),
                encode_action(Action.TURN_LEFT),
                encode_action(Action.TURN_RIGHT),
            ],
            [0, 1, 2, 3],
        )
        self.assertEqual(encode_action(INITIAL_ACTION_INDEX), -1)

    def test_resume_json_rejects_duplicate_keys(self):
        with self.assertRaisesRegex(ValueError, "duplicate key"):
            strict_json_loads('{"id": 7, "id": 8}')


class ResumeIntegrityTests(unittest.TestCase):
    def test_source_digest_is_path_free_and_binds_every_material_episode_field(self):
        source = {
            "episode_id": "episode-7",
            "trajectory_id": "trajectory-7",
            "scene_id": "/mnt/first/scenes/Rome-1.tif",
            "start_position": [0.0, 0.0, 0.0],
            "start_rotation": 0.0,
            "goals": [{"position": [1.0, 1.0, 0.0]}],
            "instruction": {"instruction_text": "stop here"},
            "reference_path": [[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
            "waypoints": [],
            "trajectory_type": "Road",
        }
        digest = episode_source_digest(7, source)
        relocated = json.loads(json.dumps(source))
        relocated["scene_id"] = "/srv/relocated/maps/Rome-1"
        self.assertEqual(episode_source_digest(7, relocated), digest)
        self.assertNotEqual(episode_source_digest(8, source), digest)
        mutations = {
            "episode_id": "episode-8",
            "trajectory_id": "trajectory-8",
            "scene_id": "Paris-1",
            "start_position": [0.1, 0.0, 0.0],
            "start_rotation": 15.0,
            "goals": [{"position": [2.0, 1.0, 0.0]}],
            "instruction": {"instruction_text": "turn left"},
            "reference_path": [[0.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
            "waypoints": [[0.5, 0.5, 0.0]],
            "trajectory_type": "Boundary",
        }
        for field, value in mutations.items():
            changed = json.loads(json.dumps(source))
            changed[field] = value
            with self.subTest(field=field):
                self.assertNotEqual(episode_source_digest(7, changed), digest)

    def test_integrity_helper_requires_relative_path_schema_and_all_frames(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir)
            annotation = make_annotation()
            rgb_dir = create_frames(output_path, annotation)

            self.assertTrue(annotation_artifacts_complete(str(output_path), annotation))

            wrong_index = dict(annotation, video="images/Rome-1_satnav_000008")
            self.assertFalse(
                annotation_artifacts_complete(str(output_path), wrong_index)
            )
            self.assertFalse(
                annotation_artifacts_complete(
                    str(output_path),
                    dict(
                        annotation,
                        _scene_id="WrongScene",
                        _episode_id="episode-7",
                    ),
                    expected_video=annotation["video"],
                    expected_scene_id="Rome-1",
                    expected_episode_id="episode-7",
                )
            )

            missing_frame = rgb_dir / "002.jpg"
            missing_frame.unlink()
            self.assertFalse(
                annotation_artifacts_complete(str(output_path), annotation)
            )
            Image.new("RGB", (2, 2)).save(missing_frame, format="JPEG")

            absolute_video = dict(annotation, video=str(output_path / "images"))
            self.assertFalse(
                annotation_artifacts_complete(str(output_path), absolute_video)
            )
            windows_absolute_video = dict(
                annotation,
                video=r"C:\private\satnav\images",
            )
            self.assertFalse(
                annotation_artifacts_complete(
                    str(output_path),
                    windows_absolute_video,
                )
            )
            escaping_video = dict(annotation, video="../outside")
            self.assertFalse(
                annotation_artifacts_complete(str(output_path), escaping_video)
            )
            bad_actions = dict(annotation, actions=[INITIAL_ACTION_INDEX])
            self.assertFalse(
                annotation_artifacts_complete(str(output_path), bad_actions)
            )
            non_terminal = dict(annotation, actions=[INITIAL_ACTION_INDEX, 1])
            self.assertFalse(
                annotation_artifacts_complete(str(output_path), non_terminal)
            )

            json_type_drift = dict(
                annotation,
                id=7.0,
                steps=1.0,
                actions=[INITIAL_ACTION_INDEX, False],
            )
            self.assertFalse(
                annotation_artifacts_complete(str(output_path), json_type_drift)
            )

            truncated_frame = rgb_dir / "001.jpg"
            Image.new("RGB", (64, 64)).save(truncated_frame, format="JPEG")
            truncated_frame.write_bytes(truncated_frame.read_bytes()[:-1])
            self.assertFalse(
                annotation_artifacts_complete(str(output_path), annotation)
            )

            (rgb_dir / "001.jpg").write_bytes(b"not-a-jpeg")
            self.assertFalse(
                annotation_artifacts_complete(str(output_path), annotation)
            )

    def test_integrity_helper_rejects_rgb_directory_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            output_path = temporary_root / "output"
            annotation = make_annotation()
            video_dir = output_path / annotation["video"]
            video_dir.mkdir(parents=True)
            outside_rgb = temporary_root / "outside" / "rgb"
            outside_rgb.mkdir(parents=True)
            for frame_index in range(1, annotation["steps"] + 2):
                (outside_rgb / f"{frame_index:03d}.jpg").touch()
            (video_dir / "rgb").symlink_to(outside_rgb, target_is_directory=True)

            self.assertFalse(
                annotation_artifacts_complete(str(output_path), annotation)
            )

    def test_integrity_helper_rejects_jpeg_symlink_escape(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            output_path = temporary_root / "output"
            annotation = make_annotation()
            rgb_dir = create_frames(output_path, annotation)
            outside_frame = temporary_root / "outside" / "secret.jpg"
            outside_frame.parent.mkdir()
            outside_frame.touch()
            first_frame = rgb_dir / "001.jpg"
            first_frame.unlink()
            first_frame.symlink_to(outside_frame)

            self.assertFalse(
                annotation_artifacts_complete(str(output_path), annotation)
            )

    def test_parallel_resume_uses_complete_cache_without_simulation(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir)
            private_scene = "/mnt/private/satnav/scenes/Rome-1"
            episode = make_episode(private_scene)
            config = make_generation_config(output_path)
            annotation = make_annotation(private_scene)
            annotation["_scene_id"] = private_scene
            annotation["_episode_id"] = "episode-7"
            annotation["_source_digest"] = episode_source_digest(7, episode)
            annotation["_generation_digest"] = episode_generation_digest(
                config, episode
            )
            create_frames(output_path, annotation)

            episode_dir, done_marker, annotation_file, _ = (
                generate_parallel._episode_paths(
                    str(output_path),
                    7,
                    private_scene,
                    "satnav",
                )
            )
            Path(episode_dir).mkdir(parents=True, exist_ok=True)
            Path(done_marker).touch()
            Path(annotation_file).write_text(
                json.dumps(annotation),
                encoding="utf-8",
            )

            environment = mock.Mock()
            with mock.patch.multiple(
                generate_parallel,
                _worker_env=environment,
                _worker_path_follower=mock.Mock(),
                _worker_dataset=SimpleNamespace(episodes=[None] * 7 + [episode]),
                _worker_output_path=str(output_path),
                _worker_dataset_name="satnav",
                _worker_config=config,
                _worker_scene_identity_cache={},
            ):
                resumed = generate_parallel.process_single_episode(7)

            environment.reset_to_episode.assert_not_called()
            self.assertEqual(resumed["_scene_id"], "Rome-1")
            rewritten_cache = json.loads(
                Path(annotation_file).read_text(encoding="utf-8")
            )
            self.assertEqual(rewritten_cache["_scene_id"], "Rome-1")
            self.assertNotIn("/mnt/private", json.dumps(rewritten_cache))

    def test_parallel_resume_regenerates_when_source_episode_changes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir)
            original_episode = make_episode(instruction="old instruction")
            current_episode = make_episode(instruction="new instruction")
            annotation = make_annotation()
            annotation["_scene_id"] = "Rome-1"
            annotation["_episode_id"] = "episode-7"
            annotation["_source_digest"] = episode_source_digest(7, original_episode)
            config = make_generation_config(output_path)
            generation_digest = episode_generation_digest(config, current_episode)
            annotation["_generation_digest"] = generation_digest
            create_frames(output_path, annotation)
            episode_dir, done_marker, annotation_file, _ = (
                generate_parallel._episode_paths(
                    str(output_path), 7, "Rome-1", "satnav"
                )
            )
            Path(done_marker).touch()
            Path(annotation_file).write_text(json.dumps(annotation), encoding="utf-8")

            environment = mock.Mock(max_episode_steps=10)
            environment.reset_to_episode.return_value = {"rgb": object()}
            with mock.patch.multiple(
                generate_parallel,
                _worker_env=environment,
                _worker_path_follower=mock.Mock(),
                _worker_dataset=SimpleNamespace(
                    episodes=[None] * 7 + [current_episode]
                ),
                _worker_output_path=str(output_path),
                _worker_dataset_name="satnav",
                _worker_config=config,
                _worker_scene_identity_cache={},
                _rollout_episode=mock.DEFAULT,
            ) as patched:
                patched["_rollout_episode"].return_value = ([-1, 0], 2, 1)
                regenerated = generate_parallel.process_single_episode(7)

            environment.reset_to_episode.assert_called_once_with(current_episode)
            self.assertEqual(
                regenerated["_source_digest"],
                episode_source_digest(7, current_episode),
            )
            self.assertNotEqual(
                regenerated["_source_digest"], annotation["_source_digest"]
            )

    def test_parallel_resume_fully_rerenders_corrupt_jpeg_cache(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir)
            episode = make_episode()
            config = make_generation_config(output_path)
            annotation = make_annotation()
            annotation["_scene_id"] = "Rome-1"
            annotation["_episode_id"] = "episode-7"
            annotation["_source_digest"] = episode_source_digest(7, episode)
            annotation["_generation_digest"] = episode_generation_digest(
                config, episode
            )
            rgb_dir = create_frames(output_path, annotation)
            (rgb_dir / "001.jpg").write_bytes(b"corrupt")
            _, done_marker, annotation_file, _ = generate_parallel._episode_paths(
                str(output_path), 7, "Rome-1", "satnav"
            )
            Path(done_marker).touch()
            Path(annotation_file).write_text(json.dumps(annotation), encoding="utf-8")
            environment = mock.Mock(max_episode_steps=10)
            environment.reset_to_episode.return_value = {"rgb": object()}

            def rerender(_env, _follower, _waypoints, _obs, target, save_images):
                self.assertTrue(save_images)
                target_path = Path(target)
                target_path.mkdir(parents=True, exist_ok=True)
                for frame_index in (1, 2):
                    Image.new("RGB", (2, 2)).save(
                        target_path / f"{frame_index:03d}.jpg", format="JPEG"
                    )
                return [-1, 0], 2, 1

            with mock.patch.multiple(
                generate_parallel,
                _worker_env=environment,
                _worker_path_follower=mock.Mock(),
                _worker_dataset=SimpleNamespace(episodes=[None] * 7 + [episode]),
                _worker_output_path=str(output_path),
                _worker_dataset_name="satnav",
                _worker_config=config,
                _worker_scene_identity_cache={},
                _rollout_episode=mock.Mock(side_effect=rerender),
            ):
                regenerated = generate_parallel.process_single_episode(7)

            environment.reset_to_episode.assert_called_once_with(episode)
            self.assertFalse(regenerated.get("_failed", False))
            self.assertTrue(
                annotation_artifacts_complete(
                    str(output_path),
                    regenerated,
                    expected_source_digest=annotation["_source_digest"],
                    expected_generation_digest=annotation["_generation_digest"],
                )
            )

    def test_serial_resume_requires_source_and_generation_identity(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_path = Path(temporary_dir)
            episode = make_episode()
            source_digest = episode_source_digest(7, episode)
            generation_digest = generation_contract_digest(None)
            annotation = make_annotation()
            annotation["source_digest"] = source_digest
            annotation["generation_digest"] = generation_digest
            annotation["scene_id"] = "Rome-1"
            annotation["episode_id"] = "episode-7"
            create_frames(output_path, annotation)
            runner = SatNavTrajectoryRunner.__new__(SatNavTrajectoryRunner)
            runner.output_path = str(output_path)

            self.assertTrue(
                runner._is_summary_entry_complete(
                    annotation,
                    expected_source_digest=source_digest,
                    expected_generation_digest=generation_digest,
                    expected_video=annotation["video"],
                    expected_scene_id="Rome-1",
                    expected_episode_id="episode-7",
                )
            )
            self.assertFalse(
                runner._is_summary_entry_complete(
                    annotation,
                    expected_source_digest="0" * 64,
                    expected_generation_digest=generation_digest,
                    expected_video=annotation["video"],
                    expected_scene_id="Rome-1",
                    expected_episode_id="episode-7",
                )
            )
            self.assertFalse(
                runner._is_summary_entry_complete(
                    annotation,
                    expected_source_digest=source_digest,
                    expected_generation_digest="1" * 64,
                    expected_video=annotation["video"],
                    expected_scene_id="Rome-1",
                    expected_episode_id="episode-7",
                )
            )

    def test_episode_generation_digest_binds_scene_content(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            config = make_generation_config(root)
            episode = make_episode()
            before = episode_generation_digest(config, episode)
            (root / "scenes" / "Rome-1.tif").write_bytes(b"changed-content")
            after = episode_generation_digest(config, episode)
            self.assertNotEqual(after, before)

    def test_resume_metadata_stays_out_of_public_annotation(self):
        annotation = make_annotation()
        annotation["_source_digest"] = "a" * 64
        annotation["_generation_digest"] = "b" * 64

        public = public_annotation(annotation)
        summary = build_summary_entry(
            annotation, scene_id="Rome-1", episode_id="episode-7"
        )

        self.assertEqual(tuple(public), PUBLIC_ANNOTATION_FIELDS)
        self.assertNotIn("_source_digest", public)
        self.assertEqual(summary["source_digest"], "a" * 64)
        self.assertEqual(summary["generation_digest"], "b" * 64)


class ReferenceFollowerConfigurationTests(unittest.TestCase):
    def test_runtime_goal_radius_updates_nested_goal_follower(self):
        follower = ReferencePathFollower(goal_radius=10.0)
        follower.goal_radius = 3.0
        follower.reset([[0.0, 0.0, 0.0], point_north(5.0)])

        self.assertEqual(
            follower.get_next_action(FakeSimulator()),
            Action.MOVE_FORWARD,
        )


if __name__ == "__main__":
    unittest.main()
