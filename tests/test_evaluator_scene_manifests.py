"""Every maintained evaluator binds selected scene bytes into its run manifest."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, Mapping

import pytest
from omegaconf import OmegaConf

from satnav.evaluation import BenchmarkManifest
from satnav.evaluation.manifest import build_run_manifest


EVALUATOR_MODULES = (
    "baselines.classic.evaluate",
    "baselines.vlm.streamvln.evaluate",
    "baselines.vlm.navila.evaluate",
    "baselines.vlm.uninavid.evaluate",
    "baselines.vlm.openfly.evaluate",
)


class _Environment:
    def close(self) -> None:
        pass


class _Policy:
    load_report: Mapping[str, Any] = {"status": "strict"}

    def close(self) -> None:
        pass


class _RecordingEvaluator:
    manifests = []

    def __init__(self, *, config, **_kwargs) -> None:
        self.config = config

    def run(self) -> Mapping[str, Any]:
        manifest = build_run_manifest(
            benchmark_digest="a" * 64,
            policy_id=self.config.policy_id,
            policy_metadata=self.config.policy_metadata,
            split=self.config.split,
            offset=self.config.offset,
            limit=self.config.limit,
            selected_keys=("val_seen::a-selected::1",),
            selected_digest="b" * 64,
            world_size=self.config.world_size,
            base_seed=self.config.base_seed,
            run_metadata={
                "evaluator": {
                    "capture_action_trace": self.config.capture_action_trace,
                    "max_steps": self.config.max_steps,
                },
                "user": self.config.run_metadata,
            },
        )
        self.manifests.append(manifest)
        return {"status": "complete"}


def _episodes(scene_root: Path):
    # Deliberately unsorted: selection must use stable keys before applying limit=1.
    return [
        SimpleNamespace(
            scene_id="z-unselected",
            episode_id="2",
            scene_path=str(scene_root / "z-unselected"),
        ),
        SimpleNamespace(
            scene_id="a-selected",
            episode_id="1",
            scene_path=str(scene_root / "a-selected"),
        ),
    ]


def _config(scene_root: Path, episodes_path: Path):
    return OmegaConf.create(
        {
            "DATASET": {
                "DATA_PATH": str(episodes_path),
                "SCENES_DIR": str(scene_root),
                "SPLIT": "val_seen",
                "VERSION": "test",
            },
            "EVAL": {"SPLIT": "val_seen"},
            "ENVIRONMENT": {"MAX_EPISODE_STEPS": 1},
            "TASK": {
                "POSSIBLE_ACTIONS": [],
                "MEASUREMENTS": [],
                "SUCCESS_DISTANCE": {"DEFAULT": 1.0},
            },
            "SIMULATOR": {
                "FORWARD_STEP_SIZE": 10.0,
                "TURN_ANGLE": 15.0,
                "RGB_SENSOR": {"WIDTH": 1, "HEIGHT": 1, "HFOV": 90.0},
            },
        }
    )


def _patch_classic(
    monkeypatch,
    module,
    active_root,
    episodes_path: Path,
) -> None:
    class _Dataset:
        def __init__(self, _dataset_config) -> None:
            self.episodes = _episodes(active_root["path"])

    class _ClassicEnvironment(_Environment):
        def __init__(self, _config, *, dataset, cycle) -> None:
            self.episodes = dataset.episodes
            self.observation_space = None
            self.action_space = None

    monkeypatch.setattr(
        module,
        "load_classic_config",
        lambda _path, overrides=(): _config(active_root["path"], episodes_path),
    )
    monkeypatch.setattr(module, "SatNavDataset", _Dataset)
    monkeypatch.setattr(module, "Env", _ClassicEnvironment)
    monkeypatch.setattr(
        module,
        "build_classic_adapter",
        lambda _method, config, **_kwargs: SimpleNamespace(
            config=config,
            adapter=_Policy(),
            policy_id="random",
            policy_metadata={"kind": "test"},
        ),
    )


def _patch_vlm(
    monkeypatch,
    module,
    active_root,
    episodes_path: Path,
) -> None:
    def build_environment(_args, _benchmark, _max_steps):
        dataset = SimpleNamespace(episodes=_episodes(active_root["path"]))
        return (
            _config(active_root["path"], episodes_path),
            dataset,
            _Environment(),
        )

    monkeypatch.setattr(module, "_build_environment", build_environment)
    baseline = module.__name__.split(".")[-2]
    if baseline == "streamvln":
        monkeypatch.setattr(
            module,
            "_checkpoint_metadata",
            lambda _args: {
                "checkpoint_id": "test",
                "checkpoint_digest": "c" * 64,
            },
        )
        monkeypatch.setattr(
            module, "_resolve_tokenizer_path", lambda _args: episodes_path.parent
        )
        monkeypatch.setattr(
            module,
            "bootstrap_streamvln",
            lambda *_args, **_kwargs: Path("."),
        )
        monkeypatch.setattr(module, "source_revision", lambda _repo: "test-revision")
        monkeypatch.setattr(
            module.StreamVLNPolicyAdapter,
            "from_pretrained",
            lambda *_args, **_kwargs: _Policy(),
        )
    elif baseline == "navila":
        monkeypatch.setattr(
            module,
            "_policy_metadata",
            lambda _args: {
                "checkpoint_id": "test",
                "checkpoint_digest": "c" * 64,
            },
        )
        monkeypatch.setattr(
            module, "bootstrap_navila", lambda *_args, **_kwargs: Path(".")
        )
        monkeypatch.setattr(module, "source_revision", lambda _repo: "test-revision")
        monkeypatch.setattr(
            module.NaVILAPolicyAdapter,
            "from_pretrained",
            lambda *_args, **_kwargs: _Policy(),
        )
    elif baseline == "uninavid":
        monkeypatch.setattr(
            module,
            "_policy_metadata",
            lambda _args: {
                "checkpoint_id": "test",
                "checkpoint_digest": "c" * 64,
            },
        )
        monkeypatch.setattr(
            module,
            "bootstrap_uninavid",
            lambda *_args, **_kwargs: Path("."),
        )
        monkeypatch.setattr(module, "source_revision", lambda _repo: "test-revision")
        monkeypatch.setattr(
            module.UniNaVidPolicyAdapter,
            "from_pretrained",
            lambda *_args, **_kwargs: _Policy(),
        )
    elif baseline == "openfly":
        monkeypatch.setattr(module, "validate_optional_upstream", lambda _path: None)
        monkeypatch.setattr(
            module,
            "openfly_model_identity",
            lambda _path: {
                "digest": "c" * 64,
                "facts": {"action_format": "compact"},
            },
        )
        monkeypatch.setattr(
            module.OpenFlyPolicyAdapter,
            "from_pretrained",
            lambda *_args, **_kwargs: _Policy(),
            raising=False,
        )
    else:  # pragma: no cover - guarded by EVALUATOR_MODULES
        raise AssertionError(f"unsupported evaluator: {baseline}")


def _arguments(
    module_name: str,
    *,
    model_path: Path,
    config_path: Path,
    task_path: Path,
    benchmark_path: Path,
    output_path: Path,
    eva_path: Path,
    processor_path: Path,
):
    if module_name == "baselines.classic.evaluate":
        return [
            "--method",
            "random",
            "--config",
            str(config_path),
            "--benchmark",
            str(benchmark_path),
            "--output-dir",
            str(output_path),
            "--split",
            "val_seen",
            "--limit",
            "1",
            "--max-steps",
            "1",
            "--device",
            "cpu",
        ]
    arguments = [
        "--model-path",
        str(model_path),
        "--task-config",
        str(task_path),
        "--benchmark-manifest",
        str(benchmark_path),
        "--output-dir",
        str(output_path),
        "--split",
        "val_seen",
        "--limit",
        "1",
        "--max-steps",
        "1",
        "--device",
        "cpu",
    ]
    if module_name == "baselines.vlm.uninavid.evaluate":
        arguments.extend(
            ["--eva-path", str(eva_path), "--processor-path", str(processor_path)]
        )
    return arguments


@pytest.mark.parametrize("module_name", EVALUATOR_MODULES)
def test_selected_scene_identity_reaches_each_evaluator_run_manifest(
    tmp_path, monkeypatch, request, module_name
):
    if (
        module_name == "baselines.vlm.openfly.evaluate"
        and "baselines.vlm.openfly.adapter" not in sys.modules
    ):
        # The core test environment intentionally has no Transformers.  Stub only
        # the model-owned adapter import; the evaluator module itself remains real.
        adapter_module = ModuleType("baselines.vlm.openfly.adapter")
        adapter_module.OpenFlyPolicyAdapter = type("OpenFlyPolicyAdapter", (), {})
        monkeypatch.setitem(
            sys.modules, "baselines.vlm.openfly.adapter", adapter_module
        )
        request.addfinalizer(lambda: sys.modules.pop(module_name, None))
    module = importlib.import_module(module_name)
    first_root = tmp_path / "first" / "scenes"
    second_root = tmp_path / "second" / "relocated-maps"
    first_root.mkdir(parents=True)
    second_root.mkdir(parents=True)
    (first_root / "a-selected.tif").write_bytes(b"same-selected-map")
    (second_root / "a-selected.tif").write_bytes(b"same-selected-map")
    # These intentionally differ: an evaluator that hashes the whole root fails.
    (first_root / "z-unselected.tif").write_bytes(b"unused-map-one")
    (second_root / "z-unselected.tif").write_bytes(b"unused-map-two")

    episodes_path = tmp_path / "all_episodes.json"
    episodes_path.write_text('{"episodes": []}\n', encoding="utf-8")
    task_path = tmp_path / "task.yaml"
    task_path.write_text("TASK: {}\n", encoding="utf-8")
    config_path = tmp_path / "classic.yaml"
    config_path.write_text("{}\n", encoding="utf-8")
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")
    model_path = tmp_path / "model"
    model_path.mkdir()
    eva_path = tmp_path / "eva.pth"
    eva_path.write_bytes(b"eva")
    processor_path = tmp_path / "processor"
    processor_path.mkdir()

    benchmark = BenchmarkManifest(
        benchmark_id="scene-manifest-test",
        dataset_version="test",
        split="val_seen",
        max_episode_steps=1,
    )
    active_root = {"path": first_root}
    monkeypatch.setattr(module, "load_benchmark_manifest", lambda _path: benchmark)
    monkeypatch.setattr(module, "Evaluator", _RecordingEvaluator)
    if module_name == "baselines.classic.evaluate":
        _patch_classic(monkeypatch, module, active_root, episodes_path)
    else:
        _patch_vlm(monkeypatch, module, active_root, episodes_path)

    _RecordingEvaluator.manifests.clear()

    def run(scene_root: Path, label: str):
        active_root["path"] = scene_root
        arguments = _arguments(
            module_name,
            model_path=model_path,
            config_path=config_path,
            task_path=task_path,
            benchmark_path=benchmark_path,
            output_path=tmp_path / f"output-{label}",
            eva_path=eva_path,
            processor_path=processor_path,
        )
        assert module.main(arguments) == 0
        manifest = _RecordingEvaluator.manifests[-1]
        return manifest["configuration"]["values"]["user"]["scenes"]

    first = run(first_root, "first")
    relocated = run(second_root, "relocated")

    assert first == relocated
    assert first["scene_count"] == 1
    assert [entry["scene_id"] for entry in first["scenes"]] == ["a-selected"]
    assert str(tmp_path) not in str(first)

    (second_root / "a-selected.tif").write_bytes(b"changed-selected-map")
    changed = run(second_root, "changed")
    assert changed["digest"] != first["digest"]
