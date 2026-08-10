import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from baselines.vlm.uninavid.trainer import (
    _install_training_audit,
    _load_yaml,
    _local_configuration,
    _validate_new_output,
    _validate_training_shape,
    build_parser as train_parser,
    build_upstream_arguments,
)


SATNAV_ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize(
    "module",
    (
        "baselines.vlm.uninavid.dataset",
        "baselines.vlm.uninavid.trainer",
        "baselines.vlm.uninavid.evaluate",
        "baselines.vlm.uninavid.checkpoint",
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
        ["bash", str(SATNAV_ROOT / "baselines/vlm/uninavid/scripts" / script), "--help"],
        cwd=SATNAV_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout.lower()


def test_local_overlay_preserves_explicit_shell_values():
    template = SATNAV_ROOT / "baselines/vlm/uninavid/local.env.example"
    result = subprocess.run(
        [
            "bash",
            "-c",
            'export UNINAVID_REPO=/explicit/repo; source "$1"; '
            'test "$UNINAVID_REPO" = /explicit/repo',
            "bash",
            str(template),
        ],
        check=False,
    )
    assert result.returncode == 0


def test_local_overlay_preserves_split_template_braces():
    template = SATNAV_ROOT / "baselines/vlm/uninavid/local.env.example"
    result = subprocess.run(
        [
            "bash",
            "-c",
            "unset SATNAV_UNINAVID_EVAL_EPISODES; source \"$1\"; "
            "test \"$SATNAV_UNINAVID_EVAL_EPISODES\" = "
            "'/absolute/path/to/SatNav-v0.1/episodes/eval/{split}/all_episodes.json'",
            "bash",
            str(template),
        ],
        check=False,
    )
    assert result.returncode == 0


def test_single_opencv_distribution_is_pinned():
    requirements = (
        SATNAV_ROOT / "baselines/vlm/uninavid/requirements.txt"
    ).read_text(encoding="utf-8")
    assert "opencv-python==4.13.0.92" in requirements
    assert "opencv-python-headless" not in requirements


def test_zero1_delegates_optimizer_and_cosine_scheduler_to_trainer():
    config_dir = SATNAV_ROOT / "baselines/vlm/uninavid/configs"
    deepspeed_config = json.loads(
        (config_dir / "zero1.json").read_text(encoding="utf-8")
    )
    train_config = _load_yaml(config_dir / "train.yaml")
    assert deepspeed_config["zero_optimization"]["stage"] == 1
    assert "optimizer" not in deepspeed_config
    assert "scheduler" not in deepspeed_config
    assert train_config["training"]["lr-scheduler-type"] == "cosine"


def test_decord_stub_has_import_spec_before_transformers_import():
    code = """
import importlib.util
from baselines.vlm.uninavid.bootstrap import install_image_dataset_decord_stub
install_image_dataset_decord_stub()
assert importlib.util.find_spec('decord') is not None
import transformers
print(transformers.__version__)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=SATNAV_ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_run_name_is_forwarded_and_only_full_training_is_supported(tmp_path):
    args = train_parser().parse_args(
        [
            "--uninavid-repo",
            str(tmp_path / "repo"),
            "--model-path",
            str(tmp_path / "model"),
            "--eva-path",
            str(tmp_path / "eva"),
            "--processor-path",
            str(tmp_path / "processor"),
            "--trajectory-root",
            str(tmp_path / "data"),
            "--data-validation-report",
            str(tmp_path / "validation.json"),
            "--output-dir",
            str(tmp_path / "output"),
            "--run-name",
            "test-run",
        ]
    )
    config = _load_yaml(SATNAV_ROOT / "baselines/vlm/uninavid/configs/train.yaml")
    upstream = list(build_upstream_arguments(args, config))
    assert upstream[upstream.index("--run_name") + 1] == "test-run"
    _validate_training_shape(args, config)
    with pytest.raises(ValueError, match="train-components=all"):
        _validate_training_shape(
            args,
            {**config, "training": {**config["training"], "train-components": "projector"}},
        )


def test_fresh_output_rejects_unknown_prior_artifacts(tmp_path):
    (tmp_path / "data_validation.json").write_text("{}", encoding="utf-8")
    _validate_new_output(tmp_path)
    (tmp_path / "pytorch_model.bin").write_bytes(b"old")
    with pytest.raises(ValueError, match="prior artifacts"):
        _validate_new_output(tmp_path)


def test_gradient_audit_requires_language_and_projector_signal(tmp_path):
    torch = pytest.importorskip("torch")

    class AuditModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language = torch.nn.Parameter(torch.ones(1))
            self.mm_projector = torch.nn.Parameter(torch.ones(1))
            self.vision_tower = torch.nn.Parameter(
                torch.ones(1), requires_grad=False
            )

    class FakeTrainer:
        def __init__(self, model):
            self.model = model
            self.state = type("State", (), {"global_step": 0})()
            self.accelerator = SimpleNamespace(sync_gradients=True)

        def training_step(self, model, inputs):
            loss = (
                model.language.sum() * inputs["language_scale"]
                + model.mm_projector.sum() * inputs["projector_scale"]
            )
            loss.backward()
            # DeepSpeed ZeRO can release these before training_step returns.
            for parameter in model.parameters():
                parameter.grad = None
            return loss.detach()

        def _save_checkpoint(self, *args, **kwargs):
            del args, kwargs

    _install_training_audit(FakeTrainer, tmp_path)
    model = AuditModel()
    trainer = FakeTrainer(model)
    with pytest.raises(RuntimeError, match="language gradient"):
        trainer.training_step(
            model,
            {
                "labels": torch.tensor([[1]]),
                "language_scale": 0.0,
                "projector_scale": 1.0,
            },
        )


def test_gradient_audit_accumulates_hooks_across_microbatches(tmp_path):
    torch = pytest.importorskip("torch")

    class AuditModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.language = torch.nn.Parameter(torch.ones(1))
            self.mm_projector = torch.nn.Parameter(torch.ones(1))
            self.vision_tower = torch.nn.Parameter(
                torch.ones(1), requires_grad=False
            )

    class FakeTrainer:
        def __init__(self, model):
            self.model = model
            self.state = SimpleNamespace(global_step=0)
            self.accelerator = SimpleNamespace(sync_gradients=False)

        def training_step(self, model, inputs):
            loss = (
                model.language.sum() * inputs["language_scale"]
                + model.mm_projector.sum() * inputs["projector_scale"]
            )
            loss.backward()
            for parameter in model.parameters():
                parameter.grad = None
            return loss.detach()

        def _save_checkpoint(self, *args, **kwargs):
            del args, kwargs

    _install_training_audit(FakeTrainer, tmp_path)
    model = AuditModel()
    trainer = FakeTrainer(model)
    trainer.training_step(
        model,
        {
            "labels": torch.tensor([[1]]),
            "language_scale": 1.0,
            "projector_scale": 0.0,
        },
    )
    assert not (tmp_path / "step_audit_rank_00000.jsonl").exists()
    trainer.accelerator.sync_gradients = True
    trainer.training_step(
        model,
        {
            "labels": torch.tensor([[1]]),
            "language_scale": 0.0,
            "projector_scale": 1.0,
        },
    )
    event = json.loads(
        (tmp_path / "step_audit_rank_00000.jsonl").read_text(encoding="utf-8")
    )
    assert event["microbatches_observed"] == 2
    assert event["nonzero_gradient"]["language"]["max_abs"] > 0
    assert event["nonzero_gradient"]["projector"]["max_abs"] > 0
    assert all(parameter.grad is None for parameter in model.parameters())


def test_train_launcher_enforces_post_training_component_delta():
    launcher = (
        SATNAV_ROOT / "baselines/vlm/uninavid/scripts/train.sh"
    ).read_text(encoding="utf-8")
    assert "--compare-model" in launcher
    assert "--expected-train-components all" in launcher
    assert "post_train_checkpoint_report.json" in launcher


def _relocated_training_configuration(root: Path, *, learning_rate=None):
    root.mkdir()
    config_path = root / "train.yaml"
    deepspeed_path = root / "zero1.json"
    config_path.write_bytes(
        (SATNAV_ROOT / "baselines/vlm/uninavid/configs/train.yaml").read_bytes()
    )
    deepspeed_path.write_bytes(
        (SATNAV_ROOT / "baselines/vlm/uninavid/configs/zero1.json").read_bytes()
    )
    command = [
        "--config",
        str(config_path),
        "--deepspeed-config",
        str(deepspeed_path),
        "--uninavid-repo",
        str(root / "upstream"),
        "--model-path",
        str(root / "model"),
        "--eva-path",
        str(root / "eva.pt"),
        "--processor-path",
        str(root / "processor"),
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
        "--video_folder",
        "--vision_tower",
        "--image_processor",
        "--output_dir",
    }
    assert path_options.isdisjoint(semantic)
    assert path_options <= set(build_upstream_arguments(first_args, first_config))
    assert path_options <= set(build_upstream_arguments(second_args, second_config))

    _, _, behavior_changed = _relocated_training_configuration(
        tmp_path / "mount-c", learning_rate="0.00002"
    )
    assert behavior_changed != first
