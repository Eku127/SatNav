"""Regression tests for production trajectory-output validation."""

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from omegaconf import OmegaConf
from PIL import Image

from applications.trajectory_generation.utils import (
    episode_generation_digest,
    episode_source_digest,
    normalize_scene_id,
)
from scripts.validation import validate_trajectory_output as validator


def _episode_dir(episode_id=0, scene_id="Rome-1"):
    return f"{scene_id}_satnav_{episode_id:06d}"


def _annotation(video=None):
    if video is None:
        video = f"images/{_episode_dir()}"
    return {
        "id": 0,
        "trajectory_id": "trajectory-0",
        "steps": 1,
        "video": video,
        "instructions": ["test"],
        "actions": [-1, 0],
    }


def _arguments(
    output_root: Path,
    annotation,
    *,
    decode_images=True,
    summary=None,
    source_episodes=None,
    expected_count=1,
):
    annotations_path = output_root / "annotations.json"
    annotations_path.write_text(json.dumps([annotation]), encoding="utf-8")
    return argparse.Namespace(
        annotations=annotations_path,
        output_root=output_root,
        summary=summary,
        source_episodes=source_episodes,
        generation_config=None,
        scenes_dir=None,
        expected_count=expected_count,
        decode_images=decode_images,
        workers=1,
    )


def _save_jpeg(path: Path, color=(10, 20, 30)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (4, 4), color=color).save(path, format="JPEG")


def _strict_arguments(
    root: Path,
    source_episodes,
    annotations,
    *,
    decode_images=False,
):
    output_root = root / "trajectory"
    scenes_dir = root / "scenes"
    scenes_dir.mkdir()
    for source_episode in source_episodes:
        scene_id = normalize_scene_id(source_episode["scene_id"])
        (scenes_dir / f"{scene_id}.tif").write_bytes(
            f"scene:{scene_id}".encode("utf-8")
        )
    for annotation in annotations:
        rgb_dir = output_root / annotation["video"] / "rgb"
        for frame_index in range(1, annotation["steps"] + 2):
            _save_jpeg(rgb_dir / f"{frame_index:03d}.jpg")
    source_path = root / "source.json"
    source_path.write_text(json.dumps({"episodes": source_episodes}), encoding="utf-8")
    config_path = root / "task.yaml"
    config_path.write_text(
        "ENVIRONMENT: {MAX_EPISODE_STEPS: 500}\n"
        "SIMULATOR:\n"
        "  TYPE: satsim\n"
        "  FORWARD_STEP_SIZE: 10\n"
        "  TURN_ANGLE: 15\n"
        "  RGB_SENSOR: {WIDTH: 448, HEIGHT: 448, HFOV: 90}\n"
        "TASK: {SUCCESS_DISTANCE: {DEFAULT: 10, Road: 10}}\n"
        f"DATASET: {{SCENES_DIR: {scenes_dir}}}\n",
        encoding="utf-8",
    )
    generation_config = OmegaConf.load(config_path)
    summary_path = output_root / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        for annotation in annotations:
            episode_id = annotation["id"]
            source_episode = source_episodes[episode_id]
            summary = {
                **annotation,
                "scene_id": normalize_scene_id(source_episode["scene_id"]),
                "episode_id": source_episode["episode_id"],
                "source_digest": episode_source_digest(episode_id, source_episode),
                "generation_digest": episode_generation_digest(
                    generation_config, source_episode
                ),
            }
            handle.write(json.dumps(summary) + "\n")
    arguments = _arguments(
        output_root,
        annotations[0],
        decode_images=decode_images,
        summary=summary_path,
        source_episodes=source_path,
        expected_count=len(source_episodes),
    )
    (output_root / "annotations.json").write_text(
        json.dumps(annotations), encoding="utf-8"
    )
    arguments.generation_config = config_path
    arguments.scenes_dir = scenes_dir
    return arguments


class TrajectoryOutputValidationTests(unittest.TestCase):
    def test_source_bound_video_directory_identity_is_canonical(self):
        source_episode = {
            "episode_id": "episode-0",
            "trajectory_id": "trajectory-0",
            "trajectory_type": "Road",
            "scene_id": "Rome-1.tif",
            "instruction": {"instruction_text": "test"},
            "goals": [{"position": [1, 1, 0]}],
            "reference_path": [[0, 0, 0], [1, 1, 0]],
        }
        invalid_videos = (
            "images/not_encoded",
            "images/WrongScene_satnav_000000",
            "images/Rome-1_satnav_000001",
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            for case_number, video in enumerate(invalid_videos):
                with self.subTest(video=video):
                    case_root = temporary_root / str(case_number)
                    case_root.mkdir()
                    arguments = _strict_arguments(
                        case_root,
                        [source_episode],
                        [_annotation(video=video)],
                    )

                    report = validator.validate(arguments)

                    self.assertEqual(report["status"], "failed")
                    self.assertTrue(
                        any(
                            "non-canonical identity" in error
                            for error in report["errors"]
                        )
                    )

            dotted_root = temporary_root / "dotted"
            dotted_root.mkdir()
            dotted_source = {**source_episode, "scene_id": "Rome.v2.tif"}
            arguments = _strict_arguments(
                dotted_root,
                [dotted_source],
                [_annotation(video="images/Rome.v2_satnav_000000")],
            )
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "passed")

    def test_source_training_identity_is_unique_and_complete(self):
        first = {
            "episode_id": "episode-0",
            "trajectory_id": "shared-trajectory",
            "trajectory_type": "Road",
            "scene_id": "Rome-1",
            "instruction": {"instruction_text": "Go   home"},
            "goals": [{"position": [1, 1, 0]}],
            "reference_path": [[0, 0, 0], [1, 1, 0]],
        }
        second = {
            **first,
            "episode_id": "episode-1",
            "instruction": {"instruction_text": "go home"},
            "reference_path": [[0, 0, 0], [2, 2, 0]],
        }
        annotations = []
        for episode_id in range(2):
            annotations.append(
                {
                    **_annotation(video=f"images/{_episode_dir(episode_id)}"),
                    "id": episode_id,
                    "trajectory_id": "shared-trajectory",
                    "instructions": [
                        first["instruction"]["instruction_text"]
                        if episode_id == 0
                        else second["instruction"]["instruction_text"]
                    ],
                }
            )
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            duplicate_root = root / "duplicate"
            duplicate_root.mkdir()
            arguments = _strict_arguments(duplicate_root, [first, second], annotations)

            report = validator.validate(arguments)

            self.assertEqual(report["status"], "failed")
            self.assertIn(
                "source episode training identity is not unique", report["errors"]
            )

            incomplete_root = root / "incomplete"
            incomplete_root.mkdir()
            incomplete = dict(first)
            incomplete.pop("trajectory_type")
            arguments = _strict_arguments(
                incomplete_root, [incomplete], [_annotation()]
            )
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "failed")
            self.assertIn("source episode 0 lacks trajectory_type", report["errors"])

    def test_decode_images_fully_loads_each_jpeg(self):
        source_episode = {
            "episode_id": "episode-0",
            "trajectory_id": "trajectory-0",
            "trajectory_type": "Road",
            "scene_id": "Rome-1",
            "instruction": {"instruction_text": "test"},
            "goals": [{"position": [1, 1, 0]}],
            "reference_path": [[0, 0, 0], [1, 1, 0]],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            arguments = _strict_arguments(
                root, [source_episode], [_annotation()], decode_images=True
            )
            frame = root / "trajectory" / "images" / _episode_dir() / "rgb" / "001.jpg"
            Image.new("RGB", (64, 64), color=(1, 2, 3)).save(frame, format="JPEG")
            frame.write_bytes(frame.read_bytes()[:-1])

            report = validator.validate(arguments)

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"]["jpeg_decode_failures"], 1)

    def test_summary_and_resume_cache_require_exact_json_types(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            episode_dir = output_root / "images" / _episode_dir()
            _save_jpeg(episode_dir / "rgb" / "001.jpg")
            _save_jpeg(episode_dir / "rgb" / "002.jpg")
            annotation = _annotation()
            summary = {
                **annotation,
                "id": 0.0,
                "steps": 1.0,
                "actions": [-1, False],
                "scene_id": "Rome-1",
                "episode_id": "episode-0",
                "source_digest": "a" * 64,
                "generation_digest": "b" * 64,
            }
            summary_path = output_root / "summary.json"
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            cache = {
                **annotation,
                "id": 0.0,
                "steps": 1.0,
                "actions": [-1, False],
                "_scene_id": "Rome-1",
                "_episode_id": "episode-0",
                "_source_digest": "a" * 64,
                "_generation_digest": "b" * 64,
            }
            (episode_dir / ".done").touch()
            (episode_dir / ".annotation.json").write_text(
                json.dumps(cache), encoding="utf-8"
            )
            arguments = _arguments(
                output_root,
                annotation,
                decode_images=False,
                summary=summary_path,
            )

            report = validator.validate(arguments)

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"]["resume_sidecar_errors"], 1)
            self.assertTrue(
                any("strict public schema" in error for error in report["errors"])
            )

    def test_release_cli_requires_nonempty_full_validation_scope(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            _save_jpeg(output_root / "images" / _episode_dir() / "rgb" / "001.jpg")
            _save_jpeg(output_root / "images" / _episode_dir() / "rgb" / "002.jpg")
            arguments = _arguments(
                output_root,
                _annotation(),
                decode_images=False,
                expected_count=-1,
            )
            arguments.report = output_root.parent / "validation.json"

            report = validator.validate(arguments)
            self.assertEqual(report["status"], "failed")
            self.assertIn("expected episode count must be positive", report["errors"])
            with self.assertRaises(ValueError):
                validator._validate_release_cli_args(arguments)

    def test_batched_decode_preserves_ordered_tree_digest(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            annotations = []
            expected_images = []
            for episode_id, steps in ((0, 2), (1, 3)):
                video = f"images/{_episode_dir(episode_id)}"
                rgb_directory = output_root / video / "rgb"
                rgb_directory.mkdir(parents=True)
                for frame_index in range(1, steps + 2):
                    image_path = rgb_directory / f"{frame_index:03d}.jpg"
                    Image.new(
                        "RGB",
                        (4, 4),
                        color=(episode_id * 50, frame_index * 20, 10),
                    ).save(image_path, format="JPEG")
                    expected_images.append(image_path)
                annotations.append(
                    {
                        "id": episode_id,
                        "trajectory_id": f"trajectory-{episode_id}",
                        "steps": steps,
                        "video": video,
                        "instructions": ["test"],
                        "actions": [-1] + [1] * steps,
                    }
                )
                annotations[-1]["actions"][-1] = 0

            annotations_path = output_root / "annotations.json"
            annotations_path.write_text(json.dumps(annotations), encoding="utf-8")
            arguments = argparse.Namespace(
                annotations=annotations_path,
                output_root=output_root,
                summary=None,
                source_episodes=None,
                generation_config=None,
                scenes_dir=None,
                expected_count=2,
                decode_images=True,
                workers=2,
            )

            with mock.patch.object(validator, "DECODE_BATCH_SIZE", 2):
                report = validator.validate(arguments)

            digest = hashlib.sha256()
            for image_path in expected_images:
                relative = str(image_path.relative_to(output_root)).encode("utf-8")
                digest.update(len(relative).to_bytes(8, "big"))
                digest.update(relative)
                digest.update(hashlib.sha256(image_path.read_bytes()).digest())
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["counts"]["jpeg_files"], 7)
            self.assertEqual(report["counts"]["jpeg_decode_failures"], 0)
            self.assertEqual(report["digests"]["image_tree_sha256"], digest.hexdigest())

    def test_rgb_directory_symlink_cannot_escape_output_root(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_root = temporary_root / "output"
            video_dir = output_root / "images" / _episode_dir()
            video_dir.mkdir(parents=True)
            outside_rgb = temporary_root / "outside" / "rgb"
            _save_jpeg(outside_rgb / "001.jpg")
            _save_jpeg(outside_rgb / "002.jpg")
            (video_dir / "rgb").symlink_to(outside_rgb, target_is_directory=True)
            arguments = _arguments(output_root, _annotation())

            with mock.patch.object(
                validator,
                "_decode_and_hash",
                wraps=validator._decode_and_hash,
            ) as decode:
                report = validator.validate(arguments)

            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any(
                    "unsafe or missing RGB directory" in error
                    for error in report["errors"]
                )
            )
            decode.assert_not_called()

    def test_jpeg_symlink_cannot_escape_output_root_or_reach_decoder(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_root = temporary_root / "output"
            rgb_dir = output_root / "images" / _episode_dir() / "rgb"
            rgb_dir.mkdir(parents=True)
            outside = temporary_root / "outside" / "secret.jpg"
            _save_jpeg(outside)
            (rgb_dir / "001.jpg").symlink_to(outside)
            _save_jpeg(rgb_dir / "002.jpg")
            arguments = _arguments(output_root, _annotation())

            with mock.patch.object(
                validator,
                "_decode_and_hash",
                wraps=validator._decode_and_hash,
            ) as decode:
                report = validator.validate(arguments)

            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any("unsafe JPEG frame 001.jpg" in error for error in report["errors"])
            )
            self.assertEqual(decode.call_count, 1)
            self.assertNotIn(str(outside), repr(decode.call_args_list))

    def test_windows_absolute_video_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            arguments = _arguments(
                output_root,
                _annotation(video=r"C:\private\satnav\images"),
                decode_images=False,
            )

            report = validator.validate(arguments)

            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any("unsafe video path" in error for error in report["errors"])
            )

    def test_unreferenced_episode_directory_is_rejected_without_reading_it(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            rgb_dir = output_root / "images" / _episode_dir() / "rgb"
            _save_jpeg(rgb_dir / "001.jpg")
            _save_jpeg(rgb_dir / "002.jpg")
            stale_dir = output_root / "images" / "stale_private_episode"
            stale_dir.mkdir()
            (stale_dir / "private-cache.bin").write_bytes(b"do not inspect")
            arguments = _arguments(output_root, _annotation(), decode_images=False)

            report = validator.validate(arguments)

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"]["unreferenced_image_directories"], 1)
            self.assertTrue(
                any(
                    "unreferenced episode directories" in error
                    for error in report["errors"]
                )
            )
            self.assertNotIn("stale_private_episode", json.dumps(report))
            self.assertNotIn("private-cache.bin", json.dumps(report))

    def test_referenced_episode_directory_rejects_unknown_files(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            episode_dir = output_root / "images" / _episode_dir()
            _save_jpeg(episode_dir / "rgb" / "001.jpg")
            _save_jpeg(episode_dir / "rgb" / "002.jpg")
            (episode_dir / ".done").touch()
            (episode_dir / ".annotation.json").write_text(
                json.dumps(_annotation()), encoding="utf-8"
            )
            (episode_dir / "private-sidecar.txt").write_text(
                "private", encoding="utf-8"
            )
            arguments = _arguments(output_root, _annotation(), decode_images=False)

            report = validator.validate(arguments)

            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"]["unexpected_image_entries"], 1)
            self.assertTrue(
                any(
                    "unexpected or unsafe entries" in error
                    for error in report["errors"]
                )
            )
            self.assertNotIn("private-sidecar.txt", json.dumps(report))

    def test_resume_sidecars_are_strictly_validated_as_an_atomic_pair(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            episode_dir = output_root / "images" / _episode_dir()
            _save_jpeg(episode_dir / "rgb" / "001.jpg")
            _save_jpeg(episode_dir / "rgb" / "002.jpg")
            annotation = _annotation()
            arguments = _arguments(output_root, annotation, decode_images=False)
            done_marker = episode_dir / ".done"
            cache_path = episode_dir / ".annotation.json"

            done_marker.write_text("/mnt/private/also-secret", encoding="utf-8")
            cache_path.write_text(
                json.dumps(
                    {
                        "private_path": "/mnt/private/secret",
                        "token": "not-validated",
                    }
                ),
                encoding="utf-8",
            )
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "failed")
            self.assertGreaterEqual(report["counts"]["resume_sidecar_errors"], 2)
            self.assertNotIn("/mnt/private", json.dumps(report))
            self.assertNotIn("not-validated", json.dumps(report))

            done_marker.write_bytes(b"")
            cache_path.write_text(
                json.dumps(
                    {
                        **annotation,
                        "_scene_id": "Rome-1",
                        "_episode_id": "episode-0",
                        "_source_digest": "a" * 64,
                        "_generation_digest": "b" * 64,
                    }
                ),
                encoding="utf-8",
            )
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["counts"]["resume_sidecar_pairs"], 1)
            self.assertEqual(report["counts"]["resume_sidecar_errors"], 0)

            done_marker.unlink()
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"]["resume_sidecar_errors"], 1)

    def test_source_episode_content_is_bound_and_report_paths_are_private(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            private_root = Path(temporary_directory) / "private-mount"
            output_root = private_root / "trajectory-output"
            rgb_dir = output_root / "images" / _episode_dir() / "rgb"
            _save_jpeg(rgb_dir / "001.jpg")
            _save_jpeg(rgb_dir / "002.jpg")
            annotation = _annotation()
            summary_path = output_root / "summary.json"
            summary_path.write_text(
                json.dumps(
                    {
                        **annotation,
                        "scene_id": "Rome-1",
                        "episode_id": "episode-0",
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            source_path = private_root / "all_episodes.json"
            source = {
                "episodes": [
                    {
                        "episode_id": "episode-0",
                        "trajectory_id": "trajectory-0",
                        "trajectory_type": "Road",
                        "scene_id": "/mnt/private/scenes/Rome-1",
                        "instruction": {"instruction_text": "changed"},
                        "reference_path": [[0, 0, 0], [1, 1, 1]],
                        "goals": [{"position": [1, 1, 1]}],
                    }
                ]
            }
            source_path.write_text(json.dumps(source), encoding="utf-8")
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            summary["source_digest"] = episode_source_digest(0, source["episodes"][0])
            summary["generation_digest"] = "b" * 64
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            arguments = _arguments(
                output_root,
                annotation,
                decode_images=False,
                summary=summary_path,
                source_episodes=source_path,
            )

            report = validator.validate(arguments)

            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any(
                    "instructions differ from source" in error
                    for error in report["errors"]
                )
            )
            rendered = json.dumps(report, sort_keys=True)
            self.assertNotIn(str(private_root), rendered)
            self.assertNotIn("/mnt/private", rendered)
            self.assertEqual(
                report["inputs"]["source_episodes"]["id"], "all_episodes.json"
            )

    def test_source_validation_discovers_summary_and_checks_all_public_fields(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_root = temporary_root / "trajectory"
            rgb_dir = output_root / "images" / _episode_dir() / "rgb"
            _save_jpeg(rgb_dir / "001.jpg")
            _save_jpeg(rgb_dir / "002.jpg")
            annotation = _annotation()
            source_path = temporary_root / "source.json"
            source = {
                "episodes": [
                    {
                        "episode_id": "episode-0",
                        "trajectory_id": "trajectory-0",
                        "trajectory_type": "Road",
                        "scene_id": "Rome-1.tif",
                        "instruction": {"instruction_text": "test"},
                        "goals": [{"position": [1, 1, 0]}],
                        "reference_path": [[0, 0, 0], [1, 1, 0]],
                    }
                ]
            }
            source_path.write_text(json.dumps(source), encoding="utf-8")
            scenes_dir = temporary_root / "scenes"
            scenes_dir.mkdir()
            scene_path = scenes_dir / "Rome-1.tif"
            scene_path.write_bytes(b"scene-v1")
            config_path = temporary_root / "task.yaml"
            config_path.write_text(
                "ENVIRONMENT:\n"
                "  MAX_EPISODE_STEPS: 500\n"
                "SIMULATOR:\n"
                "  TYPE: satsim\n"
                "  FORWARD_STEP_SIZE: 10\n"
                "  TURN_ANGLE: 15\n"
                "  RGB_SENSOR: {WIDTH: 448, HEIGHT: 448, HFOV: 90}\n"
                "TASK:\n"
                "  SUCCESS_DISTANCE: {DEFAULT: 10, Road: 10}\n"
                "DATASET:\n"
                f"  SCENES_DIR: {scenes_dir}\n",
                encoding="utf-8",
            )
            generation_config = OmegaConf.load(config_path)
            (output_root / "summary.json").write_text(
                json.dumps(
                    {
                        **annotation,
                        "scene_id": "Rome-1",
                        "episode_id": "episode-0",
                        "source_digest": episode_source_digest(
                            0, source["episodes"][0]
                        ),
                        "generation_digest": episode_generation_digest(
                            generation_config, source["episodes"][0]
                        ),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            arguments = _arguments(
                output_root,
                annotation,
                decode_images=False,
                source_episodes=source_path,
            )
            arguments.generation_config = config_path
            arguments.scenes_dir = scenes_dir

            report = validator.validate(arguments)

            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["counts"]["source_rows_matched"], 1)
            self.assertEqual(report["inputs"]["summary"]["id"], "summary.json")
            self.assertTrue(
                report["validation_scope"]["generation_contract_and_scene_content"]
            )

            scene_path.write_bytes(b"scene-v2")
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any("generation digest differs" in error for error in report["errors"])
            )
            scene_path.write_bytes(b"scene-v1")

            source["episodes"][0]["goals"] = [{"position": [2, 2, 0]}]
            source_path.write_text(json.dumps(source), encoding="utf-8")
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any("source digest differs" in error for error in report["errors"])
            )
            source["episodes"][0]["goals"] = [{"position": [1, 1, 0]}]
            source_path.write_text(json.dumps(source), encoding="utf-8")

            changed_summary = dict(annotation, actions=[-1, 1])
            changed_summary.update(
                scene_id="Rome-1",
                episode_id="episode-0",
                source_digest=episode_source_digest(0, source["episodes"][0]),
                generation_digest=episode_generation_digest(
                    generation_config, source["episodes"][0]
                ),
            )
            (output_root / "summary.json").write_text(
                json.dumps(changed_summary) + "\n", encoding="utf-8"
            )
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any("field actions differs" in error for error in report["errors"])
            )

    def test_source_validation_refuses_missing_summary(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_root = temporary_root / "trajectory"
            rgb_dir = output_root / "images" / _episode_dir() / "rgb"
            _save_jpeg(rgb_dir / "001.jpg")
            _save_jpeg(rgb_dir / "002.jpg")
            source_path = temporary_root / "source.json"
            source_path.write_text(
                json.dumps(
                    {
                        "episodes": [
                            {
                                "episode_id": "episode-0",
                                "trajectory_id": "trajectory-0",
                                "trajectory_type": "Road",
                                "scene_id": "Rome-1",
                                "instruction": {"instruction_text": "test"},
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )
            arguments = _arguments(
                output_root,
                _annotation(),
                decode_images=False,
                source_episodes=source_path,
            )

            report = validator.validate(arguments)

            self.assertEqual(report["status"], "failed")
            self.assertIn(
                "source episode validation requires a non-empty summary",
                report["errors"],
            )

    def test_public_files_and_output_root_use_exact_release_whitelists(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            rgb_dir = output_root / "images" / _episode_dir() / "rgb"
            _save_jpeg(rgb_dir / "001.jpg")
            _save_jpeg(rgb_dir / "002.jpg")
            annotation = _annotation()
            summary = {
                **annotation,
                "scene_id": "Rome-1",
                "episode_id": "episode-0",
                "source_digest": "a" * 64,
                "generation_digest": "b" * 64,
            }
            summary_path = output_root / "summary.json"
            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")

            arguments = _arguments(
                output_root,
                {**annotation, "private_path": "/mnt/private/annotation"},
                decode_images=False,
            )
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(
                any("non-public fields" in error for error in report["errors"])
            )

            arguments = _arguments(output_root, annotation, decode_images=False)
            summary_path.write_text(
                json.dumps({**summary, "private_path": "/mnt/private/summary"}) + "\n",
                encoding="utf-8",
            )
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "failed")
            self.assertTrue(any("public schema" in error for error in report["errors"]))

            summary_path.write_text(json.dumps(summary) + "\n", encoding="utf-8")
            (output_root / "private-root.txt").write_text(
                "/mnt/private/root", encoding="utf-8"
            )
            report = validator.validate(arguments)
            self.assertEqual(report["status"], "failed")
            self.assertEqual(report["counts"]["unexpected_output_root_entries"], 1)
            rendered = json.dumps(report)
            self.assertNotIn("private-root.txt", rendered)
            self.assertNotIn("/mnt/private", rendered)

    def test_duplicate_json_keys_are_rejected_without_echoing_values(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            output_root = Path(temporary_directory)
            rgb_dir = output_root / "images" / _episode_dir() / "rgb"
            _save_jpeg(rgb_dir / "001.jpg")
            _save_jpeg(rgb_dir / "002.jpg")
            annotation = _annotation()
            arguments = _arguments(output_root, annotation, decode_images=False)
            annotations_path = output_root / "annotations.json"
            safe_field = '"trajectory_id": "trajectory-0"'
            annotations_path.write_text(
                json.dumps([annotation]).replace(
                    safe_field,
                    '"trajectory_id": "/mnt/private/hidden", ' + safe_field,
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(ValueError, "duplicate key") as captured:
                validator.validate(arguments)
            self.assertNotIn("/mnt/private", str(captured.exception))

            annotations_path.write_text(json.dumps([annotation]), encoding="utf-8")
            summary = {
                **annotation,
                "scene_id": "Rome-1",
                "episode_id": "episode-0",
                "source_digest": "a" * 64,
                "generation_digest": "b" * 64,
            }
            summary_path = output_root / "summary.json"
            safe_scene = '"scene_id": "Rome-1"'
            summary_path.write_text(
                json.dumps(summary).replace(
                    safe_scene,
                    '"scene_id": "/mnt/private/hidden", ' + safe_scene,
                )
                + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate key") as captured:
                validator.validate(arguments)
            self.assertNotIn("/mnt/private", str(captured.exception))

    def test_report_destination_cannot_mutate_any_validated_input(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_root = temporary_root / "trajectory"
            rgb_dir = output_root / "images" / _episode_dir() / "rgb"
            _save_jpeg(rgb_dir / "001.jpg")
            _save_jpeg(rgb_dir / "002.jpg")
            annotation = _annotation()
            arguments = _arguments(output_root, annotation, decode_images=False)
            source_path = temporary_root / "source.json"
            source_path.write_text('{"episodes": []}\n', encoding="utf-8")
            config_path = temporary_root / "task.yaml"
            config_path.write_text("TASK: {}\n", encoding="utf-8")
            scenes_dir = temporary_root / "scenes"
            scenes_dir.mkdir()
            scene_path = scenes_dir / "Rome-1.tif"
            scene_path.write_bytes(b"scene")
            arguments.source_episodes = source_path
            arguments.generation_config = config_path
            arguments.scenes_dir = scenes_dir

            protected_targets = (
                output_root / "validation.json",
                output_root / "annotations.json",
                rgb_dir / "001.jpg",
                source_path,
                config_path,
                scene_path,
            )
            for target in protected_targets:
                with self.subTest(target=target.name):
                    arguments.report = target
                    with self.assertRaises(ValueError):
                        validator._validated_report_path(arguments)

            arguments.report = temporary_root / "reports" / "validation.json"
            resolved = validator._validated_report_path(arguments)
            self.assertEqual(
                resolved,
                temporary_root / "reports" / "validation.json",
            )

    def test_output_root_and_public_manifests_cannot_escape_through_symlinks(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            output_root = temporary_root / "trajectory"
            rgb_dir = output_root / "images" / _episode_dir() / "rgb"
            _save_jpeg(rgb_dir / "001.jpg")
            _save_jpeg(rgb_dir / "002.jpg")
            annotation = _annotation()
            arguments = _arguments(output_root, annotation, decode_images=False)

            output_alias = temporary_root / "trajectory-alias"
            output_alias.symlink_to(output_root, target_is_directory=True)
            arguments.output_root = output_alias
            arguments.annotations = output_alias / "annotations.json"
            arguments.report = temporary_root / "validation.json"
            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                validator.validate(arguments)
            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                validator._validated_report_path(arguments)

            arguments.output_root = output_root
            external_annotations = temporary_root / "external-annotations.json"
            external_annotations.write_text(json.dumps([annotation]), encoding="utf-8")
            arguments.annotations = external_annotations
            with self.assertRaisesRegex(
                ValueError, "annotations must be output_root/annotations.json"
            ):
                validator.validate(arguments)

            arguments.annotations = output_root / "annotations.json"
            external_summary = temporary_root / "external-summary.json"
            external_summary.write_text("", encoding="utf-8")
            arguments.summary = external_summary
            with self.assertRaisesRegex(
                ValueError, "summary must be output_root/summary.json"
            ):
                validator.validate(arguments)

            arguments.summary = None
            annotations_path = output_root / "annotations.json"
            annotations_path.unlink()
            external_annotations.write_text(json.dumps([annotation]), encoding="utf-8")
            annotations_path.symlink_to(external_annotations)
            with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                validator.validate(arguments)


if __name__ == "__main__":
    unittest.main()
