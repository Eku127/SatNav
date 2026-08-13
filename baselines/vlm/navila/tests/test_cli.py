import json
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.vlm.navila.artifacts import navila_model_identity
from baselines.vlm.navila.evaluate import build_parser as eval_parser
from baselines.vlm.navila.trainer import (
    _load_yaml,
    _local_configuration,
    _training_payload,
    build_parser as train_parser,
    build_upstream_arguments,
)
from satnav.training.manifest import ManifestMismatchError, ensure_manifest


SATNAV_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "module",
    (
        "baselines.vlm.navila.dataset",
        "baselines.vlm.navila.trainer",
        "baselines.vlm.navila.evaluate",
        "baselines.vlm.navila.checkpoint",
    ),
)
def test_python_cli_help_needs_no_weights_or_upstream(module):
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=SATNAV_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


@pytest.mark.parametrize(
    "script", ("download.sh", "train.sh", "eval.sh", "patch_environment.sh")
)
def test_shell_cli_help_needs_no_weights_or_upstream(script):
    if script == "patch_environment.sh":
        return
    result = subprocess.run(
        ["bash", str(SATNAV_ROOT / "baselines/vlm/navila/scripts" / script), "--help"],
        cwd=SATNAV_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def _fake_model(root: Path) -> Path:
    model = root / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    for component in ("llm", "mm_projector", "vision_tower"):
        path = model / component
        path.mkdir()
        (path / "config.json").write_text("{}\n", encoding="utf-8")
        (path / "model.safetensors").write_bytes(component.encode())
    (model / "llm" / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (model / "llm" / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (model / "vision_tower" / "preprocessor_config.json").write_text(
        "{}\n", encoding="utf-8"
    )
    return model


def test_model_identity_covers_all_components_and_tokenizer(tmp_path):
    identity = navila_model_identity(_fake_model(tmp_path))
    assert identity["scope"] == "complete-navila-composite-checkpoint"
    assert set(identity["components"]) == {"llm", "mm_projector", "vision_tower"}
    llm_files = {entry["path"] for entry in identity["components"]["llm"]["files"]}
    assert {"model.safetensors", "tokenizer.json", "tokenizer_config.json"} <= llm_files


def test_zero2_uses_torch_adam_without_host_cuda_jit():
    config = json.loads(
        (SATNAV_ROOT / "baselines/vlm/navila/configs/zero2.json").read_text()
    )
    assert config["zero_optimization"]["stage"] == 2
    assert config["optimizer"]["params"]["torch_adam"] is True


def test_parsers_keep_heavy_imports_out_of_help_path(tmp_path):
    eval_args = eval_parser().parse_args(["--model-path", str(tmp_path / "model")])
    assert eval_args.num_frames == 8
    train_args = train_parser().parse_args(
        [
            "--model-path",
            str(tmp_path / "model"),
            "--trajectory-root",
            str(tmp_path / "data"),
            "--data-validation-report",
            str(tmp_path / "validation.json"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert train_args.report_to == "none"


def test_training_identity_rejects_world_size_change(tmp_path, monkeypatch):
    model = _fake_model(tmp_path)
    trajectory = tmp_path / "trajectory"
    rgb = trajectory / "images" / "sample" / "rgb"
    rgb.mkdir(parents=True)
    for index in range(1, 3):
        (rgb / f"{index:03d}.jpg").write_bytes(b"jpeg")
    annotations = [
        {
            "id": 1,
            "trajectory_id": "t1",
            "steps": 1,
            "video": "images/sample",
            "instructions": ["stop"],
            "actions": [-1, 0],
        }
    ]
    annotation_path = trajectory / "annotations.json"
    annotation_path.write_text(json.dumps(annotations), encoding="utf-8")
    from baselines.vlm.navila.dataset import load_annotation_records, validate_records

    _, records = load_annotation_records(trajectory)
    validation = dict(
        validate_records(
            annotation_path,
            records,
            strict_frames=True,
            decode_samples=0,
            hash_frame_content=True,
        ),
        allow_external_paths=False,
    )
    report = tmp_path / "validation.json"
    report.write_text(json.dumps(validation), encoding="utf-8")
    args = train_parser().parse_args(
        [
            "--model-path",
            str(model),
            "--trajectory-root",
            str(trajectory),
            "--data-validation-report",
            str(report),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    config = {"training": {"per-device-batch-size": 1}}
    manifest = tmp_path / "training_manifest.json"
    monkeypatch.setenv("WORLD_SIZE", "2")
    ensure_manifest(manifest, _training_payload(args, config, "pinned"))
    monkeypatch.setenv("WORLD_SIZE", "4")
    with pytest.raises(ManifestMismatchError, match="different manifest"):
        ensure_manifest(manifest, _training_payload(args, config, "pinned"))


def _relocated_training_configuration(root: Path, *, learning_rate=None):
    root.mkdir()
    config_path = root / "train.yaml"
    deepspeed_path = root / "zero2.json"
    config_path.write_bytes(
        (SATNAV_ROOT / "baselines/vlm/navila/configs/train.yaml").read_bytes()
    )
    deepspeed_path.write_bytes(
        (SATNAV_ROOT / "baselines/vlm/navila/configs/zero2.json").read_bytes()
    )
    command = [
        "--config",
        str(config_path),
        "--deepspeed-config",
        str(deepspeed_path),
        "--model-path",
        str(root / "model"),
        "--trajectory-root",
        str(root / "trajectory"),
        "--data-validation-report",
        str(root / "validation.json"),
        "--output-dir",
        str(root / "output"),
    ]
    if learning_rate is not None:
        command.extend(("--learning-rate", learning_rate))
    args = train_parser().parse_args(command)
    config = _load_yaml(config_path)
    return args, config, _local_configuration(args, config)


def test_training_configuration_is_path_free_and_relocation_stable(tmp_path):
    first_args, first_config, first = _relocated_training_configuration(
        tmp_path / "mount-a"
    )
    second_args, second_config, second = _relocated_training_configuration(
        tmp_path / "mount-b"
    )

    assert first == second
    serialized = json.dumps(first, sort_keys=True)
    assert str(tmp_path) not in serialized
    assert "upstream_arguments" not in first
    semantic = first["semantic_upstream_arguments"]
    path_options = {
        "--deepspeed",
        "--model_name_or_path",
        "--data_path",
        "--image_folder",
        "--vision_tower",
        "--output_dir",
    }
    assert path_options.isdisjoint(semantic)
    assert path_options <= set(build_upstream_arguments(first_args, first_config))
    assert path_options <= set(build_upstream_arguments(second_args, second_config))

    _, _, behavior_changed = _relocated_training_configuration(
        tmp_path / "mount-c", learning_rate="0.00004"
    )
    assert behavior_changed != first
