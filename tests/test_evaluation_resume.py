"""Fake-env coverage for JSONL, resume, faults, errors, and aggregation."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pytest

from satnav.evaluation import (
    BenchmarkManifest,
    EvaluationConfig,
    Evaluator,
    ManifestMismatchError,
    PolicyStep,
    ResultValidationError,
    aggregate_run,
)


@dataclass
class FakeEpisode:
    scene_id: str
    episode_id: str
    trajectory_id: str = "trajectory"
    trajectory_type: str = "Road"


@dataclass
class FakeAgentState:
    position: list
    rotation: float


class FakeEnv:
    max_episode_steps = 4

    def __init__(self, episodes):
        self.episodes = list(episodes)
        self.current_episode = None
        self.episode_over = False
        self.last_step_info = {}
        self._steps = 0
        self.closed = False

    @property
    def simulator(self):
        return self

    @property
    def agent_state(self):
        return FakeAgentState([float(self._steps), 0.0, 0.0], 0.0)

    def reset_to_episode(self, episode):
        self.current_episode = episode
        self.episode_over = False
        self.last_step_info = {}
        self._steps = 0
        return {"episode_id": episode.episode_id, "step": 0}

    def step(self, action):
        self._steps += 1
        self.episode_over = int(action) == 0 or self._steps >= self.max_episode_steps
        self.last_step_info = {
            "elapsed_steps": self._steps,
            "terminated_by": "stop" if int(action) == 0 else "max_steps",
        }
        return (
            {"episode_id": self.current_episode.episode_id, "step": self._steps},
            self.episode_over,
            self.last_step_info,
        )

    def get_metrics(self):
        return {
            "distance_to_goal": float(max(0, 2 - self._steps)),
            "success": float(self._steps >= 2),
            "spl": float(self._steps >= 2),
            "path_length": float(self._steps),
            # Non-scalar metric must not inflate a JSONL result.
            "top_down_map": {"map": [[0] * 8 for _ in range(8)]},
        }

    def close(self):
        self.closed = True


class FakePolicy:
    def __init__(self, fail_episode_id=None):
        self.fail_episode_id = fail_episode_id
        self.context = None
        self.local_step = 0
        self.reset_keys = []
        self.seeds = {}
        self.closed = False

    def reset(self, context):
        self.context = context
        self.local_step = 0
        self.reset_keys.append(context.episode_key)
        self.seeds[context.episode_key] = context.seed

    def act(self, observation: Mapping[str, Any]):
        if self.context.episode.episode_id == self.fail_episode_id:
            raise RuntimeError("injected policy failure")
        self.local_step += 1
        return PolicyStep(
            action=1 if self.local_step == 1 else 0,
            info={"local_step": self.local_step},
        )

    def close(self):
        self.closed = True


class FakeVideoHook:
    def __init__(self):
        self.events = []
        self.closed = False

    def on_episode_start(self, context, observation):
        self.events.append(("start", context.episode_key, observation["step"]))

    def on_step(self, context, step_index, policy_step, observation, info):
        self.events.append(("step", context.episode_key, step_index))

    def on_episode_end(self, context, record):
        self.events.append(("end", context.episode_key, record["status"]))

    def close(self):
        self.closed = True


def benchmark():
    return BenchmarkManifest(
        benchmark_id="fake-v1",
        dataset_version="fake-1",
        split="test",
        kind="smoke",
        action_space=("STOP", "MOVE_FORWARD"),
        required_metrics=("success", "spl", "distance_to_goal", "path_length"),
    )


def episodes():
    # Deliberately unsorted, with bare ID reuse across scenes.
    return [
        FakeEpisode("scene-b", "1"),
        FakeEpisode("scene-a", "2"),
        FakeEpisode("scene-a", "1"),
        FakeEpisode("scene-c", "0"),
        FakeEpisode("scene-b", "0"),
    ]


def run_rank(
    output_dir: Path,
    *,
    rank=0,
    world_size=1,
    resume=False,
    fault_injector=None,
    fail_episode_id=None,
    base_seed=17,
):
    env = FakeEnv(episodes())
    policy = FakePolicy(fail_episode_id=fail_episode_id)
    evaluator = Evaluator(
        environment=env,
        policy=policy,
        config=EvaluationConfig(
            output_dir=output_dir,
            split="test",
            policy_id="fake-policy",
            rank=rank,
            world_size=world_size,
            base_seed=base_seed,
            resume=resume,
            aggregate_single_rank=True,
            policy_metadata={"checkpoint_sha256": "abc"},
        ),
        benchmark=benchmark(),
        fault_injector=fault_injector,
    )
    result = evaluator.run()
    assert env.closed
    assert policy.closed
    return result, policy


def read_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def test_single_and_two_rank_runs_have_same_union_and_traces(tmp_path):
    single_dir = tmp_path / "single"
    multi_dir = tmp_path / "multi"
    single_summary, _ = run_rank(single_dir)
    run_rank(multi_dir, rank=0, world_size=2)
    run_rank(multi_dir, rank=1, world_size=2)
    multi_summary = aggregate_run(multi_dir)

    single_records = read_jsonl(single_dir / "rank_00000" / "episodes.jsonl")
    multi_records = read_jsonl(multi_dir / "rank_00000" / "episodes.jsonl")
    multi_records += read_jsonl(multi_dir / "rank_00001" / "episodes.jsonl")
    single_by_key = {record["episode_key"]: record for record in single_records}
    multi_by_key = {record["episode_key"]: record for record in multi_records}

    assert set(single_by_key) == set(multi_by_key)
    for key in single_by_key:
        assert single_by_key[key]["seed"] == multi_by_key[key]["seed"]
        assert single_by_key[key]["action_trace"] == multi_by_key[key]["action_trace"]
        assert single_by_key[key]["metrics"] == multi_by_key[key]["metrics"]
    assert single_summary["metrics"] == multi_summary["metrics"]
    assert multi_summary["status"] == "complete"
    assert multi_summary["validation"]["duplicate_keys"] == []
    assert multi_summary["validation"]["missing_keys"] == []


def test_fault_after_record_resumes_without_repeating_completed_episode(tmp_path):
    output_dir = tmp_path / "resume"
    injected = {"raised": False}

    def interrupt_after_first_record(stage, context, record):
        if stage == "after_record" and not injected["raised"]:
            injected["raised"] = True
            raise RuntimeError("simulated worker interruption")

    with pytest.raises(RuntimeError, match="simulated worker interruption"):
        run_rank(output_dir, fault_injector=interrupt_after_first_record)
    assert not (output_dir / "rank_00000" / "done.json").exists()
    first_records = read_jsonl(output_dir / "rank_00000" / "episodes.jsonl")
    assert len(first_records) == 1

    summary, resumed_policy = run_rank(output_dir, resume=True)
    assert summary["status"] == "complete"
    assert len(resumed_policy.reset_keys) == len(episodes()) - 1
    assert first_records[0]["episode_key"] not in resumed_policy.reset_keys
    assert len(read_jsonl(output_dir / "rank_00000" / "episodes.jsonl")) == len(
        episodes()
    )


def test_manifest_mismatch_refuses_resume_before_rollout(tmp_path):
    output_dir = tmp_path / "mismatch"
    run_rank(output_dir)

    env = FakeEnv(episodes())
    policy = FakePolicy()
    evaluator = Evaluator(
        environment=env,
        policy=policy,
        config=EvaluationConfig(
            output_dir=output_dir,
            split="test",
            policy_id="fake-policy",
            base_seed=999,
            resume=True,
            policy_metadata={"checkpoint_sha256": "abc"},
        ),
        benchmark=benchmark(),
    )
    with pytest.raises(ManifestMismatchError, match="different manifest"):
        evaluator.run()
    assert policy.reset_keys == []
    assert env.closed and policy.closed


def test_episode_exception_is_an_explicit_error_record(tmp_path):
    output_dir = tmp_path / "episode-error"
    summary, _ = run_rank(output_dir, fail_episode_id="2")

    assert summary["status"] == "completed_with_errors"
    assert summary["error_episode_count"] == 1
    records = read_jsonl(output_dir / "rank_00000" / "episodes.jsonl")
    error_record = next(record for record in records if record["status"] == "error")
    assert error_record["error"]["type"] == "RuntimeError"
    assert error_record["error"] == {
        "type": "RuntimeError",
        "code": "episode_evaluation_error",
        "message": "Episode evaluation failed.",
    }
    assert (output_dir / "rank_00000" / "done.json").exists()


def test_episode_error_record_does_not_persist_private_paths_or_traceback(tmp_path):
    output_dir = tmp_path / "private-error"
    source_root = Path(__file__).resolve().parents[1]
    private_root = Path("/mnt/private/satnav-checkpoints")

    def raise_private_path_error(stage, context, record):
        del context, record
        if stage == "before_episode":
            raise FileNotFoundError(
                f"{private_root}/model.bin; workspace={source_root}; tmp={tmp_path}"
            )

    summary, _ = run_rank(output_dir, fault_injector=raise_private_path_error)

    assert summary["status"] == "completed_with_errors"
    assert summary["error_episode_count"] == len(episodes())
    records_path = output_dir / "rank_00000" / "episodes.jsonl"
    records = read_jsonl(records_path)
    assert all(record["status"] == "error" for record in records)
    assert all(
        record["error"]
        == {
            "type": "FileNotFoundError",
            "code": "episode_evaluation_error",
            "message": "Episode evaluation failed.",
        }
        for record in records
    )

    serialized = records_path.read_text(encoding="utf-8")
    assert str(private_root) not in serialized
    assert str(tmp_path) not in serialized
    assert str(source_root) not in serialized
    assert "traceback" not in serialized.lower()


def test_optional_video_hook_observes_rollout_without_core_video_imports(tmp_path):
    output_dir = tmp_path / "video-hook"
    env = FakeEnv(episodes()[:1])
    policy = FakePolicy()
    hook = FakeVideoHook()
    evaluator = Evaluator(
        environment=env,
        policy=policy,
        config=EvaluationConfig(
            output_dir=output_dir,
            split="test",
            policy_id="fake-policy",
        ),
        benchmark=BenchmarkManifest(
            benchmark_id="video-fake",
            dataset_version="fake-1",
            split="test",
            required_metrics=("success",),
        ),
        video_hook=hook,
    )

    summary = evaluator.run()
    assert summary["status"] == "complete"
    assert [event[0] for event in hook.events] == ["start", "step", "step", "end"]
    assert hook.closed


def test_aggregator_reports_duplicate_and_missing_records(tmp_path):
    output_dir = tmp_path / "invalid"
    run_rank(output_dir)
    records_path = output_dir / "rank_00000" / "episodes.jsonl"
    records = read_jsonl(records_path)
    # Replace the last expected record with a duplicate of the first.  The
    # existing done marker makes this a useful post-run integrity check.
    with records_path.open("w", encoding="utf-8") as handle:
        for record in records[:-1] + [records[0]]:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    with pytest.raises(ResultValidationError) as raised:
        aggregate_run(output_dir)
    validation = raised.value.summary["validation"]
    assert validation["duplicate_keys"] == [records[0]["episode_key"]]
    assert validation["missing_keys"] == [records[-1]["episode_key"]]
