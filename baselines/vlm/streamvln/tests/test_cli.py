import json
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.vlm.streamvln.evaluate import (
    _checkpoint_metadata,
    _episode_artifact_identity,
    _run_metadata,
    build_parser as eval_parser,
)
from baselines.vlm.streamvln.trainer import (
    _model_path_with_tokenizer_hint,
    _validate_smoke_sample_cap,
    build_parser as train_parser,
)
from satnav.evaluation.manifest import (
    ManifestMismatchError,
    build_run_manifest,
    ensure_manifest,
)


SATNAV_ROOT = Path(__file__).resolve().parents[4]


def test_python_parsers_accept_minimal_paths_without_loading_weights(tmp_path):
    parsed_eval = eval_parser().parse_args(["--model-path", str(tmp_path / "model")])
    assert parsed_eval.num_frames == 32
    assert parsed_eval.num_history == 8
    assert parsed_eval.num_future_steps == 4

    parsed_train = train_parser().parse_args(
        [
            "--model-path", str(tmp_path / "model"),
            "--vision-tower", "tower",
            "--trajectory-root", str(tmp_path / "data"),
            "--output-dir", str(tmp_path / "output"),
        ]
    )
    assert parsed_train.report_to is None
    assert parsed_train.data_augmentation is None
    assert parsed_train.torch_compile is None


@pytest.mark.parametrize(
    "module",
    (
        "baselines.vlm.streamvln.dataset",
        "baselines.vlm.streamvln.trainer",
        "baselines.vlm.streamvln.evaluate",
        "baselines.vlm.streamvln.scripts.normalize_decord_wheel",
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


@pytest.mark.parametrize("script", ("download.sh", "train.sh", "eval.sh"))
def test_shell_cli_help_needs_no_weights_or_upstream(script):
    result = subprocess.run(
        ["bash", str(SATNAV_ROOT / "baselines/vlm/streamvln/scripts" / script), "--help"],
        cwd=SATNAV_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_common_shell_propagates_configured_python_bin_to_path(tmp_path):
    fake_bin = tmp_path / "fresh-env" / "bin"
    fake_bin.mkdir(parents=True)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    common = SATNAV_ROOT / "baselines/vlm/streamvln/scripts/_common.sh"
    command = (
        f'source "{common}"; '
        f'STREAMVLN_PYTHON="{fake_python}"; '
        "streamvln_prepare_python; "
        'test "$STREAMVLN_PYTHON_BIN" = "$STREAMVLN_PYTHON"; '
        'test "${PATH%%:*}" = "$(dirname "$STREAMVLN_PYTHON")"'
    )
    result = subprocess.run(
        ["bash", "-c", command],
        cwd=SATNAV_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_zero2_uses_torch_adamw_without_host_cuda_jit_dependency():
    config_path = SATNAV_ROOT / "baselines/vlm/streamvln/configs/zero2.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    assert config["zero_optimization"]["stage"] == 2
    assert config["optimizer"]["type"] == "AdamW"
    assert config["optimizer"]["params"]["torch_adam"] is True


def test_explicit_sample_cap_must_fill_global_microbatch(tmp_path, monkeypatch):
    args = train_parser().parse_args(
        [
            "--model-path", str(tmp_path / "model"),
            "--vision-tower", "tower",
            "--trajectory-root", str(tmp_path / "data"),
            "--output-dir", str(tmp_path / "output"),
            "--per-device-batch-size", "1",
        ]
    )
    config = {"training": {"dataloader_drop_last": True}}
    monkeypatch.setenv("WORLD_SIZE", "4")
    monkeypatch.setenv("SATNAV_MAX_SAMPLES", "1")
    with pytest.raises(ValueError, match="global microbatch"):
        _validate_smoke_sample_cap(args, config)
    monkeypatch.setenv("SATNAV_MAX_SAMPLES", "4")
    _validate_smoke_sample_cap(args, config)


def _identity_args(tmp_path):
    model = tmp_path / "model"
    tokenizer = tmp_path / "tokenizer"
    tower = tmp_path / "tower"
    model.mkdir()
    tokenizer.mkdir()
    tower.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    (model / "generation_config.json").write_text("{}\n", encoding="utf-8")
    (model / "model-00001-of-00002.safetensors").write_bytes(b"weights-v1-a")
    (model / "model-00002-of-00002.safetensors").write_bytes(b"weights-v1-b")
    (model / "model.safetensors.index.json").write_text(
        '{"weight_map":{"layer.a":"model-00001-of-00002.safetensors",'
        '"layer.b":"model-00002-of-00002.safetensors"}}\n',
        encoding="utf-8",
    )
    (tokenizer / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
    (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    (tower / "config.json").write_text("{}\n", encoding="utf-8")
    (tower / "model.safetensors").write_bytes(b"tower-v1")
    (tower / "preprocessor_config.json").write_text(
        '{"size":384}\n', encoding="utf-8"
    )
    task_config = tmp_path / "task.yaml"
    task_config.write_text("TASK: v1\n", encoding="utf-8")
    args = eval_parser().parse_args(
        [
            "--model-path",
            str(model),
            "--tokenizer-path",
            str(tokenizer),
            "--vision-tower",
            str(tower),
            "--task-config",
            str(task_config),
        ]
    )
    return args


def _identity_run_payload(args, benchmark_path):
    return build_run_manifest(
        benchmark_digest="a" * 64,
        policy_id="streamvln:test",
        policy_metadata=_checkpoint_metadata(args),
        split="val_seen",
        offset=0,
        limit=1,
        selected_keys=("val_seen::scene::episode",),
        selected_digest="b" * 64,
        world_size=1,
        base_seed=0,
        run_metadata={"user": _run_metadata(args, benchmark_path)},
    )


@pytest.mark.parametrize(
    "changed_fact",
    (
        "checkpoint",
        "tokenizer",
        "vision_tower",
        "max_new_tokens",
        "dtype",
        "attention_implementation",
        "task_config",
    ),
)
def test_streamvln_resume_identity_rejects_material_changes(tmp_path, changed_fact):
    args = _identity_args(tmp_path)
    benchmark_path = tmp_path / "benchmark.json"
    benchmark_path.write_text("{}\n", encoding="utf-8")
    manifest_path = tmp_path / "run_manifest.json"
    ensure_manifest(manifest_path, _identity_run_payload(args, benchmark_path))

    if changed_fact == "checkpoint":
        (args.model_path / "model-00002-of-00002.safetensors").write_bytes(
            b"weights-v2-b"
        )
    elif changed_fact == "tokenizer":
        (args.tokenizer_path / "tokenizer.json").write_text(
            '{"changed":true}\n', encoding="utf-8"
        )
    elif changed_fact == "vision_tower":
        (Path(args.vision_tower) / "model.safetensors").write_bytes(b"tower-v2")
    elif changed_fact == "max_new_tokens":
        args.max_new_tokens += 1
    elif changed_fact == "dtype":
        args.dtype = "float16"
    elif changed_fact == "attention_implementation":
        args.attention_implementation = "sdpa"
    elif changed_fact == "task_config":
        args.task_config.write_text("TASK: v2\n", encoding="utf-8")

    with pytest.raises(ManifestMismatchError, match="different manifest"):
        ensure_manifest(manifest_path, _identity_run_payload(args, benchmark_path))


def test_checkpoint_identity_covers_all_weight_shards(tmp_path):
    args = _identity_args(tmp_path)
    metadata = _checkpoint_metadata(args)
    identity = metadata["checkpoint_identity"]
    assert identity["scope"] == "from-pretrained-model-and-config-artifacts"
    assert {entry["path"] for entry in identity["files"]} == {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    assert {
        entry["path"] for entry in metadata["vision_tower_identity"]["files"]
    } == {"config.json", "model.safetensors", "preprocessor_config.json"}
    assert (
        metadata["vision_tower_identity"]["scope"]
        == "complete-local-vision-tower-snapshot"
    )
    assert metadata["max_new_tokens"] == 10000
    assert metadata["dtype"] == "bfloat16"
    assert metadata["attention_implementation"] == "flash_attention_2"


def test_episode_artifact_identity_rejects_content_change(tmp_path):
    episodes = tmp_path / "episodes.json"
    episodes.write_text('{"episodes":[]}\n', encoding="utf-8")
    identity = _episode_artifact_identity(episodes, None)
    assert identity["id"] == "episodes.json"
    assert identity["size"] == episodes.stat().st_size
    assert len(identity["sha256"]) == 64

    episodes.write_text('{"episodes":[{"episode_id":"changed"}]}\n', encoding="utf-8")
    with pytest.raises(ManifestMismatchError, match="dataset_digest"):
        _episode_artifact_identity(episodes, identity["sha256"])


def test_renamed_local_checkpoint_keeps_qwen_tokenizer_selection(tmp_path):
    import copy

    model = tmp_path / "renamed-checkpoint"
    model.mkdir()
    (model / "tokenizer_config.json").write_text(
        '{"tokenizer_class":"Qwen2Tokenizer"}\n', encoding="utf-8"
    )
    hinted = _model_path_with_tokenizer_hint(model)
    assert str(hinted) == str(model)
    assert "qwen" in hinted.lower()
    copied = copy.deepcopy(hinted)
    assert str(copied) == str(model)
    assert "qwen" in copied.lower()
