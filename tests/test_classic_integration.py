"""CPU-only integration coverage for the four classic evaluator adapters."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from unittest import mock

import numpy as np
import pytest
from omegaconf import OmegaConf

from baselines.classic.common.config import load_classic_config
from baselines.classic.factory import build_classic_adapter
from satnav.evaluation import EpisodeContext, PolicyStep


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class FakeState:
    position: list
    rotation: float = 0.0


class FakeSimulator:
    def get_agent_state(self):
        return FakeState([0.0, 0.0, 0.0])


class FakeEnvironment:
    simulator = FakeSimulator()

    @property
    def agent_state(self):
        return self.simulator.get_agent_state()


class FakeEpisode:
    episode_id = "1"
    scene_id = "scene"
    start_position = [0.0, 0.0, 0.0]
    reference_path = [[0.0, 0.0, 0.0], [0.001, 0.0, 0.0]]
    goals = []

    def __init__(self, trajectory_type="Road"):
        self.trajectory_type = trajectory_type


def context(trajectory_type="Road"):
    return EpisodeContext(
        episode=FakeEpisode(trajectory_type),
        episode_key=f"test::scene::{trajectory_type}",
        split="test",
        episode_index=0,
        rank=0,
        world_size=1,
        max_steps=5,
        seed=7,
        environment=FakeEnvironment(),
    )


def test_factory_import_and_nonlearning_build_are_torch_free():
    code = """
import sys
from omegaconf import OmegaConf
from baselines.classic.factory import build_classic_adapter
assert 'torch' not in sys.modules
cfg = OmegaConf.create({'MODEL': {'RANDOM_AGENT': {'action_probs': [0.1, 0.6, 0.2, 0.1]}}})
bundle = build_classic_adapter('random', cfg)
assert bundle.policy_id == 'random'
assert 'torch' not in sys.modules
"""
    environment = {"PYTHONDONTWRITEBYTECODE": "1", "PYTHONPATH": str(REPOSITORY_ROOT)}
    subprocess.run(
        [sys.executable, "-B", "-c", code],
        cwd=str(REPOSITORY_ROOT),
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )


def test_reference_factory_uses_evaluation_specific_success_distance():
    config = OmegaConf.create(
        {
            "SIMULATOR": {"TURN_ANGLE": 15},
            "TASK": {
                "SUCCESS_DISTANCE": {
                    "DEFAULT": 10,
                    "Boundary": 10,
                    "LandmarkSet": 30,
                    "Road": 10,
                }
            },
        }
    )
    adapter = build_classic_adapter("reference", config).adapter

    adapter.reset(context("LandmarkSet"))
    assert adapter.act({}).info["goal_radius"] == 30.0
    adapter.reset(context("Road"))
    assert adapter.act({}).info["goal_radius"] == 10.0


def test_config_loader_honors_task_path_override_before_task_merge():
    config = load_classic_config(
        REPOSITORY_ROOT / "configs/baselines/seq2seq_eval.yaml",
        overrides=(
            "BASE_TASK_CONFIG_PATH=configs/satnav_eval_task.yaml",
            "DATASET.SPLIT=val_unseen",
        ),
    )

    assert config.SIMULATOR.RGB_SENSOR.WIDTH == 448
    assert config.DATASET.SPLIT == "val_unseen"
    assert config.IL.lr == pytest.approx(3e-4)
    assert "_base_" not in config


def test_classic_cli_resolves_config_without_loading_torch(capsys):
    from baselines.classic.evaluate import main

    result = main(
        [
            "--method",
            "reference_follower",
            "--print-config",
            "--set",
            "SIMULATOR.TURN_ANGLE=30",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert result == 0
    assert payload["method"] == "reference_follower"
    assert payload["config"]["SIMULATOR"]["TURN_ANGLE"] == 30


def test_shell_entrypoints_are_valid_bash():
    for path in (
        REPOSITORY_ROOT / "scripts/classic/eval.sh",
        REPOSITORY_ROOT / "scripts/classic/eval_parallel.sh",
        REPOSITORY_ROOT / "scripts/classic/eval_checkpoint.sh",
        REPOSITORY_ROOT / "scripts/classic/eval_checkpoint_parallel.sh",
        REPOSITORY_ROOT / "scripts/seq2seq/eval.sh",
        REPOSITORY_ROOT / "scripts/seq2seq/eval_parallel.sh",
        REPOSITORY_ROOT / "scripts/cma/eval.sh",
        REPOSITORY_ROOT / "scripts/cma/eval_parallel.sh",
    ):
        subprocess.run(["bash", "-n", str(path)], check=True)


def test_legacy_eval_scripts_delegate_to_generic_classic_entrypoint():
    for path in (
        REPOSITORY_ROOT / "scripts/seq2seq/eval.sh",
        REPOSITORY_ROOT / "scripts/seq2seq/eval_parallel.sh",
        REPOSITORY_ROOT / "scripts/cma/eval.sh",
        REPOSITORY_ROOT / "scripts/cma/eval_parallel.sh",
    ):
        source = path.read_text(encoding="utf-8")
        assert "../classic/" in source
        assert "run.py" not in source
        assert "eval_ckpt" not in source


def test_single_rank_classic_entrypoints_honor_cuda_devices():
    for path in (
        REPOSITORY_ROOT / "scripts/classic/eval.sh",
        REPOSITORY_ROOT / "scripts/classic/eval_checkpoint.sh",
    ):
        source = path.read_text(encoding="utf-8")
        assert "CUDA_DEVICES=" in source
        assert 'export CUDA_VISIBLE_DEVICES="${CUDA_DEVICES}"' in source


@pytest.mark.parametrize(
    "script,arguments,extra_environment",
    (
        ("eval_parallel.sh", ("random", "val_seen", "2", "2,3", "1"), {}),
        (
            "eval_checkpoint_parallel.sh",
            ("seq2seq", "experiment", "val_seen", "2", "2,3", "1"),
            {"SEQ2SEQ_CUDA_DEVICES": "6,7"},
        ),
    ),
)
def test_parallel_classic_entrypoints_isolate_each_rank_gpu(
    tmp_path, script, arguments, extra_environment
):
    fake_bin = tmp_path / "bin"
    fake_bin.mkdir()
    child_log = tmp_path / "children.jsonl"
    fake_bash = fake_bin / "bash"
    fake_bash.write_text(
        "#!/bin/sh\n"
        "python3 -c 'import json, os; print(json.dumps({k: os.environ.get(k) "
        "for k in (\"CUDA_VISIBLE_DEVICES\", \"CUDA_DEVICES\", \"SATNAV_RANK\")}))' "
        '>> "$SATNAV_TEST_CHILD_LOG"\n',
        encoding="utf-8",
    )
    fake_bash.chmod(0o755)
    fake_python = fake_bin / "python"
    fake_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake_python.chmod(0o755)
    environment = {
        "PATH": f"{fake_bin}:{os.environ['PATH']}",
        "HOME": str(tmp_path / "home"),
        "SATNAV_CONDA_SH": str(tmp_path / "missing-conda.sh"),
        "SATNAV_CLASSIC_OUTPUT": str(tmp_path / "output"),
        "OUTPUT_ROOT": str(tmp_path / "output"),
        "SATNAV_TEST_CHILD_LOG": str(child_log),
        "CUDA_DEVICES": "4,5",
        **extra_environment,
    }
    subprocess.run(
        ["/bin/bash", str(REPOSITORY_ROOT / "scripts/classic" / script), *arguments],
        cwd=REPOSITORY_ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    children = [json.loads(line) for line in child_log.read_text().splitlines()]
    assert {
        (child["SATNAV_RANK"], child["CUDA_VISIBLE_DEVICES"], child["CUDA_DEVICES"])
        for child in children
    } == {("0", "2", "2"), ("1", "3", "3")}


def test_legacy_evaluator_compatibility_path_uses_public_env_api():
    source = (
        REPOSITORY_ROOT / "satnav/training/evaluator.py"
    ).read_text(encoding="utf-8")
    assert "env._task" not in source
    assert "env._sim" not in source
    assert "env.agent_state" in source
    assert 'getattr(env, "last_step_info"' in source


def test_legacy_policy_imports_and_explicit_factories_remain_identical():
    from baselines.classic.cma import CMAPolicy as NewCMA
    from baselines.classic.cma.factory import build_cma_adapter
    from baselines.classic.seq2seq import Seq2SeqPolicy as NewSeq2Seq
    from baselines.classic.seq2seq.factory import build_seq2seq_adapter
    from satnav.models.baselines.cma_policy import CMAPolicy as LegacyCMA
    from satnav.models.baselines.seq2seq_policy import (
        Seq2SeqPolicy as LegacySeq2Seq,
    )

    assert NewSeq2Seq is LegacySeq2Seq
    assert NewCMA is LegacyCMA
    assert callable(build_seq2seq_adapter)
    assert callable(build_cma_adapter)


def test_explicit_factories_forward_the_legacy_policy_classes():
    from baselines.classic.cma import factory as cma_factory
    from baselines.classic.seq2seq import factory as seq2seq_factory
    from satnav.models.baselines.cma_policy import CMAPolicy
    from satnav.models.baselines.seq2seq_policy import Seq2SeqPolicy

    sentinel = object()
    common = {
        "config": OmegaConf.create({}),
        "observation_space": {},
        "action_space": {"actions": ["STOP"]},
        "checkpoint_path": "checkpoint.pth",
        "vocab_path": "vocab.json",
    }
    with mock.patch.object(
        seq2seq_factory, "build_torch_il_adapter", return_value=sentinel
    ) as seq_builder:
        assert seq2seq_factory.build_seq2seq_adapter(**common) is sentinel
        assert seq_builder.call_args.kwargs["policy_class"] is Seq2SeqPolicy
        assert seq_builder.call_args.kwargs["method"] == "seq2seq"
    with mock.patch.object(
        cma_factory, "build_torch_il_adapter", return_value=sentinel
    ) as cma_builder:
        assert cma_factory.build_cma_adapter(**common) is sentinel
        assert cma_builder.call_args.kwargs["policy_class"] is CMAPolicy
        assert cma_builder.call_args.kwargs["method"] == "cma"


def test_strict_checkpoint_loads_model_and_optimizer(tmp_path):
    import torch

    from baselines.classic.common.checkpoints import load_training_checkpoint

    source = torch.nn.Linear(3, 2)
    optimizer = torch.optim.Adam(source.parameters(), lr=1e-3)
    checkpoint = {
        "config": {"MODEL": {"policy_name": "seq2seq"}},
        "epoch": 2,
        "loss": 0.5,
        "optim_state": optimizer.state_dict(),
        "state_dict": source.state_dict(),
        "step_id": 9,
    }
    path = tmp_path / "valid.pth"
    torch.save(checkpoint, path)
    target = torch.nn.Linear(3, 2)
    target_optimizer = torch.optim.Adam(target.parameters(), lr=9e-3)

    loaded = load_training_checkpoint(
        str(path), target, optimizer=target_optimizer, device="cpu"
    )
    assert loaded["step_id"] == 9
    for expected, actual in zip(source.parameters(), target.parameters()):
        assert torch.equal(expected, actual)


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda checkpoint: checkpoint.pop("optim_state"), "missing keys"),
        (lambda checkpoint: checkpoint.update(step_id=0), "no optimizer step"),
        (lambda checkpoint: checkpoint.update(loss=float("nan")), "not finite"),
    ],
)
def test_strict_checkpoint_rejects_incomplete_artifacts(
    tmp_path, mutation, match
):
    import torch

    from baselines.classic.common.checkpoints import read_training_checkpoint

    checkpoint = {
        "config": {},
        "epoch": 0,
        "loss": 1.0,
        "optim_state": {},
        "state_dict": {},
        "step_id": 1,
    }
    mutation(checkpoint)
    path = tmp_path / "invalid.pth"
    torch.save(checkpoint, path)
    with pytest.raises(ValueError, match=match):
        read_training_checkpoint(str(path))


def test_torch_adapter_suppresses_early_stop_and_batches_numpy():
    import torch

    from baselines.classic.common.torch_policy_adapter import TorchILPolicyAdapter

    class Net:
        @staticmethod
        def get_initial_state(batch_size, device):
            return torch.zeros(1, batch_size, 2, device=device)

    class Policy(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = Net()
            self.seen = None

        def act(self, observations, state, previous, mask, deterministic=True):
            self.seen = observations
            return torch.zeros(1, 1, dtype=torch.long), state

    policy = Policy()
    adapter = TorchILPolicyAdapter(policy, device="cpu", min_stop_steps=1)
    adapter.reset(context())
    step = adapter.act({"rgb": np.zeros((2, 2, 3), dtype=np.uint8)})
    assert isinstance(step, PolicyStep)
    assert step.action == "MOVE_FORWARD"
    assert tuple(policy.seen["rgb"].shape) == (1, 2, 2, 3)
