import json
import subprocess
import sys
from pathlib import Path

import pytest

from baselines.vlm.streamvln.evaluate import build_parser as eval_parser
from baselines.vlm.streamvln.trainer import (
    _model_path_with_tokenizer_hint,
    _validate_smoke_sample_cap,
    build_parser as train_parser,
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
