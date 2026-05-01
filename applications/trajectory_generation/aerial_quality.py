from __future__ import annotations

"""Shared AerialSim trajectory quality scoring utilities."""

from typing import Mapping, Optional


SCORE_FORMULA = "avg_std + 0.25*avg_height_range + 0.02*avg_mean - 120.0*avg_black_frac"


def compute_aerial_quality_score(
    *,
    avg_std: float,
    avg_height_range: float,
    avg_mean: float,
    avg_black_frac: float,
) -> float:
    """Score AerialSim frames by texture, 3D height variation, and bad pixels."""
    return float(
        avg_std
        + 0.25 * avg_height_range
        + 0.02 * avg_mean
        - 120.0 * avg_black_frac
    )


def compute_aerial_quality_score_from_row(row: Mapping[str, object]) -> float:
    """Compute the shared AerialSim quality score from a metrics row."""
    return compute_aerial_quality_score(
        avg_std=float(row.get("avg_std", 0.0)),
        avg_height_range=float(row.get("avg_height_range", 0.0)),
        avg_mean=float(row.get("avg_mean", 0.0)),
        avg_black_frac=float(row.get("avg_black_frac", 0.0)),
    )


def passes_aerial_quality_thresholds(
    row: Mapping[str, object],
    *,
    min_avg_std: float,
    max_avg_black_frac: float,
    min_avg_height_range: float,
    min_score: Optional[float] = None,
) -> bool:
    """Return whether a metrics row passes the shared AerialSim thresholds."""
    avg_std = float(row.get("avg_std", float("-inf")))
    avg_black_frac = float(row.get("avg_black_frac", float("inf")))
    avg_height_range = float(row.get("avg_height_range", float("-inf")))
    ok = (
        avg_std >= float(min_avg_std)
        and avg_black_frac <= float(max_avg_black_frac)
        and avg_height_range >= float(min_avg_height_range)
    )
    if min_score is not None:
        score = float(row.get("score", float("-inf")))
        ok = ok and score >= float(min_score)
    return bool(ok)


def format_aerial_quality_threshold_rule(
    *,
    min_avg_std: float,
    max_avg_black_frac: float,
    min_avg_height_range: float,
    min_score: Optional[float] = None,
) -> str:
    """Format threshold settings for reports."""
    rule = (
        f"avg_std>={min_avg_std} && avg_black_frac<={max_avg_black_frac} && "
        f"avg_height_range>={min_avg_height_range}"
    )
    if min_score is not None:
        rule += f" && score>={min_score}"
    return rule
