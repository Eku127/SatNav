"""Public, framework-independent SatNav evaluation API."""

from satnav.evaluation.adapter import EpisodeContext, PolicyAdapter, PolicyStep
from satnav.evaluation.artifacts import referenced_scene_identity
from satnav.evaluation.episodes import (
    EpisodePlan,
    EpisodeSelectionError,
    PlannedEpisode,
    build_episode_plan,
    episode_keys_digest,
    episode_seed,
    stable_episode_key,
)
from satnav.evaluation.evaluator import (
    EpisodeEvaluationError,
    EvaluationConfig,
    EvaluationError,
    Evaluator,
    FaultInjector,
    VideoHook,
)
from satnav.evaluation.manifest import (
    BenchmarkManifest,
    ManifestError,
    ManifestMismatchError,
    load_benchmark_manifest,
    payload_digest,
)
from satnav.evaluation.results import (
    RankResultStore,
    ResultError,
    ResultValidationError,
    ResumeRequiredError,
    aggregate_run,
    rank_directory,
)

__all__ = [
    "BenchmarkManifest",
    "EpisodeContext",
    "EpisodeEvaluationError",
    "EpisodePlan",
    "EpisodeSelectionError",
    "EvaluationConfig",
    "EvaluationError",
    "Evaluator",
    "FaultInjector",
    "ManifestError",
    "ManifestMismatchError",
    "PlannedEpisode",
    "PolicyAdapter",
    "PolicyStep",
    "RankResultStore",
    "ResultError",
    "ResultValidationError",
    "ResumeRequiredError",
    "VideoHook",
    "aggregate_run",
    "build_episode_plan",
    "episode_keys_digest",
    "episode_seed",
    "load_benchmark_manifest",
    "payload_digest",
    "rank_directory",
    "referenced_scene_identity",
    "stable_episode_key",
]
