"""Framework-agnostic SatNav rollout evaluator.

This module intentionally imports neither Torch nor any SatNav trainer.  Model
setup, checkpoint loading, tokenization, recurrent state, and action chunking
belong in ``PolicyAdapter`` implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional, Protocol, Sequence

from satnav.evaluation._json import to_jsonable
from satnav.evaluation.adapter import EpisodeContext, PolicyAdapter, PolicyStep
from satnav.evaluation.episodes import (
    EpisodePlan,
    PlannedEpisode,
    build_episode_plan,
    episode_seed,
)
from satnav.evaluation.results import (
    RESULT_SCHEMA_VERSION,
    RankResultStore,
    aggregate_run,
)


class EvaluationError(RuntimeError):
    """Base class for evaluator-level failures."""


class EpisodeEvaluationError(EvaluationError):
    """Raised after persisting an episode error when fail-fast is enabled."""


class VideoHook(Protocol):
    """Optional observer for video generation outside the evaluator core."""

    def on_episode_start(
        self, context: EpisodeContext, observation: Mapping[str, Any]
    ) -> None:
        """Start collecting frames for an episode."""

    def on_step(
        self,
        context: EpisodeContext,
        step_index: int,
        policy_step: PolicyStep,
        observation: Mapping[str, Any],
        info: Mapping[str, Any],
    ) -> None:
        """Observe one completed environment step."""

    def on_episode_end(
        self, context: EpisodeContext, record: Mapping[str, Any]
    ) -> None:
        """Finalize video output for an episode."""

    def close(self) -> None:
        """Release hook-owned resources."""


FaultInjector = Callable[
    [str, EpisodeContext, Optional[Mapping[str, Any]]], None
]


@dataclass(frozen=True)
class EvaluationConfig:
    """Immutable facts controlling one evaluation run."""

    output_dir: Path
    split: str
    policy_id: str
    offset: int = 0
    limit: Optional[int] = None
    rank: int = 0
    world_size: int = 1
    base_seed: int = 0
    max_steps: Optional[int] = None
    resume: bool = False
    fail_fast: bool = False
    fail_on_episode_error: bool = False
    capture_action_trace: bool = True
    aggregate_single_rank: bool = True


def _scalar_metrics(metrics: Mapping[str, Any]) -> Mapping[str, Any]:
    """Keep scalar metrics and omit map/image payloads from JSONL records."""

    result: Dict[str, Any] = {}
    for name, value in metrics.items():
        try:
            normalized = to_jsonable(value)
        except (TypeError, ValueError):
            continue
        if normalized is None or isinstance(normalized, (str, bool, int, float)):
            result[str(name)] = normalized
    return result


def _episode_metadata(episode: Any) -> Mapping[str, Any]:
    fields = ("scene_id", "episode_id", "trajectory_id", "trajectory_type")
    return {
        name: to_jsonable(getattr(episode, name, None))
        for name in fields
        if getattr(episode, name, None) is not None
    }


def _safe_agent_state(environment: Any) -> Optional[Mapping[str, Any]]:
    try:
        return to_jsonable(environment.agent_state)
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return None


class Evaluator:
    """Run one policy adapter over a deterministic rank-local episode shard."""

    def __init__(
        self,
        *,
        environment: Any,
        policy: PolicyAdapter,
        config: EvaluationConfig,
        video_hook: Optional[VideoHook] = None,
        fault_injector: Optional[FaultInjector] = None,
    ) -> None:
        self.environment = environment
        self.policy = policy
        self.config = config
        self.video_hook = video_hook
        self.fault_injector = fault_injector

    def _max_steps(self) -> int:
        configured = self.config.max_steps
        value = (
            configured
            if configured is not None
            else getattr(self.environment, "max_episode_steps", None)
        )
        if value is None or int(value) <= 0:
            raise EvaluationError(
                "max_steps must be positive in EvaluationConfig or environment"
            )
        return int(value)

    def _episodes(self, episodes: Optional[Sequence[Any]]) -> Sequence[Any]:
        if episodes is not None:
            return tuple(episodes)
        try:
            return tuple(self.environment.episodes)
        except AttributeError as error:
            raise EvaluationError(
                "Environment must expose public episodes or run() must receive them"
            ) from error

    def _context(
        self, planned: PlannedEpisode, max_steps: int
    ) -> EpisodeContext:
        return EpisodeContext(
            episode=planned.episode,
            episode_key=planned.key,
            split=self.config.split,
            episode_index=planned.episode_index,
            rank=self.config.rank,
            world_size=self.config.world_size,
            max_steps=max_steps,
            seed=episode_seed(self.config.base_seed, planned.key),
            environment=self.environment,
        )

    def _inject(
        self,
        stage: str,
        context: EpisodeContext,
        record: Optional[Mapping[str, Any]] = None,
    ) -> None:
        if self.fault_injector is not None:
            self.fault_injector(stage, context, record)

    def _evaluate_episode(
        self, planned: PlannedEpisode, context: EpisodeContext
    ) -> Mapping[str, Any]:
        self._inject("before_episode", context)
        observation = self.environment.reset_to_episode(planned.episode)
        if not isinstance(observation, Mapping):
            raise EvaluationError("environment.reset_to_episode() must return a mapping")
        initial_metrics = _scalar_metrics(self.environment.get_metrics())
        initial_agent_state = _safe_agent_state(self.environment)
        self.policy.reset(context)
        if self.video_hook is not None:
            self.video_hook.on_episode_start(context, observation)

        action_trace = []
        last_info: Mapping[str, Any] = {}
        terminated_by = "max_steps"
        steps_executed = 0
        for step_index in range(context.max_steps):
            if bool(getattr(self.environment, "episode_over", False)):
                terminated_by = "environment"
                break
            policy_step = self.policy.act(observation)
            if not isinstance(policy_step, PolicyStep):
                raise EvaluationError(
                    "PolicyAdapter.act() must return satnav.evaluation.PolicyStep"
                )
            transition = self.environment.step(policy_step.action)
            if not isinstance(transition, tuple) or len(transition) != 3:
                raise EvaluationError(
                    "environment.step() must return (observation, done, info)"
                )
            observation, done, info = transition
            steps_executed += 1
            if not isinstance(observation, Mapping):
                raise EvaluationError("environment.step() observation must be a mapping")
            if info is None:
                info = getattr(self.environment, "last_step_info", {}) or {}
            if not isinstance(info, Mapping):
                raise EvaluationError("environment.step() info must be a mapping")
            last_info = info
            trace_step = {
                "step": step_index,
                "action": to_jsonable(policy_step.action),
                "policy_info": to_jsonable(policy_step.info),
            }
            if self.config.capture_action_trace:
                action_trace.append(trace_step)
            if self.video_hook is not None:
                self.video_hook.on_step(
                    context, step_index, policy_step, observation, info
                )
            if bool(done) or bool(getattr(self.environment, "episode_over", False)):
                terminated_by = str(
                    info.get("terminated_by")
                    or info.get("termination_reason")
                    or "environment"
                )
                break

        metrics = _scalar_metrics(self.environment.get_metrics())
        steps_taken = int(
            last_info.get("elapsed_steps", steps_executed)
            if last_info
            else steps_executed
        )
        record: Mapping[str, Any] = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "episode_key": planned.key,
            "episode_index": planned.episode_index,
            "rank": self.config.rank,
            "status": "ok",
            "seed": context.seed,
            "episode": _episode_metadata(planned.episode),
            "steps_taken": steps_taken,
            "terminated_by": terminated_by,
            "initial_metrics": initial_metrics,
            "metrics": metrics,
            "initial_agent_state": initial_agent_state,
            "final_agent_state": _safe_agent_state(self.environment),
            "action_trace": action_trace,
        }
        if self.video_hook is not None:
            self.video_hook.on_episode_end(context, record)
        return record

    def _error_record(
        self,
        planned: PlannedEpisode,
        context: EpisodeContext,
        error: Exception,
    ) -> Mapping[str, Any]:
        return {
            "schema_version": RESULT_SCHEMA_VERSION,
            "episode_key": planned.key,
            "episode_index": planned.episode_index,
            "rank": self.config.rank,
            "status": "error",
            "seed": context.seed,
            "episode": _episode_metadata(planned.episode),
            "error": {
                "type": type(error).__name__,
                "code": "episode_evaluation_error",
                "message": "Episode evaluation failed.",
            },
            "metrics": {},
            "steps_taken": 0,
            "action_trace": [],
        }

    def _close(self) -> None:
        first_error: Optional[Exception] = None
        for resource in (self.video_hook, self.policy, self.environment):
            if resource is None:
                continue
            close = getattr(resource, "close", None)
            if not callable(close):
                continue
            try:
                close()
            except Exception as error:  # pragma: no cover - defensive cleanup
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise EvaluationError(f"Failed to close evaluation resource: {first_error}")

    def _run_open(
        self, episodes: Optional[Sequence[Any]] = None
    ) -> Mapping[str, Any]:
        all_episodes = self._episodes(episodes)
        max_steps = self._max_steps()
        plan: EpisodePlan = build_episode_plan(
            all_episodes,
            split=self.config.split,
            offset=self.config.offset,
            limit=self.config.limit,
            rank=self.config.rank,
            world_size=self.config.world_size,
        )
        output_dir = Path(self.config.output_dir)
        store = RankResultStore(output_dir, self.config.rank)

        records = list(store.prepare(resume=self.config.resume))
        completed = {
            str(record.get("episode_key"))
            for record in records
            if record.get("episode_key") is not None
        }

        for planned in plan.shard:
            if planned.key in completed:
                continue
            context = self._context(planned, max_steps)
            try:
                record = self._evaluate_episode(planned, context)
            except Exception as error:
                record = self._error_record(planned, context, error)
            record = dict(record)
            store.append(record)
            records.append(record)
            completed.add(planned.key)
            # This stage is deliberately outside the episode try/except so
            # tests and operators can simulate abrupt worker interruption.
            self._inject("after_record", context, record)
            if record["status"] == "error" and self.config.fail_fast:
                raise EpisodeEvaluationError(
                    f"Episode failed: {planned.key}; error record was persisted"
                )

        done = store.mark_done(expected_keys=plan.shard_keys, records=records)
        if self.config.world_size == 1 and self.config.aggregate_single_rank:
            return aggregate_run(
                output_dir,
                fail_on_episode_error=self.config.fail_on_episode_error,
            )
        return {
            "status": done["status"],
            "rank": self.config.rank,
            "world_size": self.config.world_size,
            "expected_count": len(plan.shard_keys),
            "record_count": len(records),
            "error_count": done["error_count"],
        }

    def run(
        self, episodes: Optional[Sequence[Any]] = None
    ) -> Mapping[str, Any]:
        """Evaluate the local shard, persist its done marker, and return status."""

        try:
            result = self._run_open(episodes)
        except BaseException:
            # Never hide the actual rollout/interruption error behind a cleanup
            # failure.  A clean run still treats close failures as errors.
            try:
                self._close()
            except Exception:
                pass
            raise
        else:
            self._close()
            return result
