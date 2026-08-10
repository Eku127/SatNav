"""Deterministic episode identity, selection, and strided sharding."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence, Tuple


KEY_SEPARATOR = "::"


class EpisodeSelectionError(ValueError):
    """Raised when episodes cannot form an unambiguous evaluation plan."""


def _identity_component(value: Any, name: str) -> str:
    component = str(value)
    if not component:
        raise EpisodeSelectionError(f"Episode {name} must be non-empty")
    if KEY_SEPARATOR in component:
        raise EpisodeSelectionError(
            f"Episode {name} contains reserved separator {KEY_SEPARATOR!r}: "
            f"{component!r}"
        )
    return component


def stable_episode_key(split: str, episode: Any) -> str:
    """Return ``split::scene_id::episode_id`` for an episode.

    The dataset loader is responsible for keeping ``scene_id`` as a logical
    identifier rather than converting it to a host-specific absolute path.
    """

    return KEY_SEPARATOR.join(
        (
            _identity_component(split, "split"),
            _identity_component(getattr(episode, "scene_id", ""), "scene_id"),
            _identity_component(getattr(episode, "episode_id", ""), "episode_id"),
        )
    )


def episode_keys_digest(keys: Iterable[str]) -> str:
    """Hash an ordered episode-key sequence using a canonical JSON-free form."""

    digest = hashlib.sha256()
    for key in keys:
        encoded = key.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, byteorder="big", signed=False))
        digest.update(encoded)
    return digest.hexdigest()


def episode_seed(base_seed: int, episode_key: str) -> int:
    """Derive a deterministic unsigned 64-bit seed for one episode."""

    payload = f"{int(base_seed)}\0{episode_key}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


@dataclass(frozen=True)
class PlannedEpisode:
    """An episode plus its index in the selected, pre-sharding sequence."""

    episode: Any
    key: str
    episode_index: int


@dataclass(frozen=True)
class EpisodePlan:
    """Global selection and the deterministic rank-local strided shard."""

    split: str
    offset: int
    limit: Optional[int]
    rank: int
    world_size: int
    selected: Tuple[PlannedEpisode, ...]
    shard: Tuple[PlannedEpisode, ...]

    @property
    def selected_keys(self) -> Tuple[str, ...]:
        return tuple(item.key for item in self.selected)

    @property
    def shard_keys(self) -> Tuple[str, ...]:
        return tuple(item.key for item in self.shard)

    @property
    def selected_digest(self) -> str:
        return episode_keys_digest(self.selected_keys)

    @property
    def shard_digest(self) -> str:
        return episode_keys_digest(self.shard_keys)


def build_episode_plan(
    episodes: Sequence[Any],
    *,
    split: str,
    offset: int = 0,
    limit: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> EpisodePlan:
    """Sort, slice, then stride-shard episodes deterministically.

    Ordering is by stable key, ``offset`` and ``limit`` apply to that global
    ordering, and rank ``r`` receives ``selected[r::world_size]``.  A limit of
    ``None`` or ``-1`` means all remaining episodes; ``0`` means no episodes.
    """

    offset = int(offset)
    rank = int(rank)
    world_size = int(world_size)
    if offset < 0:
        raise EpisodeSelectionError(f"offset must be >= 0, got {offset}")
    if world_size <= 0:
        raise EpisodeSelectionError(
            f"world_size must be greater than zero, got {world_size}"
        )
    if rank < 0 or rank >= world_size:
        raise EpisodeSelectionError(
            f"rank must be in [0, {world_size}), got {rank}"
        )
    if limit is not None:
        limit = int(limit)
        if limit < -1:
            raise EpisodeSelectionError(f"limit must be -1 or >= 0, got {limit}")
        if limit == -1:
            limit = None

    keyed = sorted(
        ((stable_episode_key(split, episode), episode) for episode in episodes),
        key=lambda item: item[0],
    )
    duplicate_keys = [
        key for index, (key, _) in enumerate(keyed[1:], start=1)
        if key == keyed[index - 1][0]
    ]
    if duplicate_keys:
        rendered = ", ".join(sorted(set(duplicate_keys))[:5])
        raise EpisodeSelectionError(
            "Stable episode keys must be unique; duplicate key(s): " + rendered
        )

    stop = None if limit is None else offset + limit
    sliced = keyed[offset:stop]
    selected = tuple(
        PlannedEpisode(episode=episode, key=key, episode_index=index)
        for index, (key, episode) in enumerate(sliced)
    )
    shard = selected[rank::world_size]
    return EpisodePlan(
        split=_identity_component(split, "split"),
        offset=offset,
        limit=limit,
        rank=rank,
        world_size=world_size,
        selected=selected,
        shard=shard,
    )
