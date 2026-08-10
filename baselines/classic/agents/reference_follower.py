"""Reference-path follower adapter for the generic evaluator."""

from __future__ import annotations

from typing import Any, List, Mapping, Optional, Sequence

from satnav.evaluation import EpisodeContext, PolicyStep
from satnav.navigation import ReferencePathFollower


def _position(value: Any) -> Optional[List[float]]:
    """Normalize a dataset waypoint without changing its stored schema."""
    if isinstance(value, Mapping):
        value = value.get("position")
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [float(component) for component in value]
    return None


class ReferenceFollowerAdapter:
    """Follow the episode's reference path using only public Env contracts."""

    def __init__(
        self,
        goal_radius: float = 3.0,
        turn_angle: float = 15.0,
        success_distances: Optional[Mapping[str, float]] = None,
    ) -> None:
        self.goal_radius = float(goal_radius)
        self.turn_angle = float(turn_angle)
        self.success_distances = {
            str(name): float(value)
            for name, value in (success_distances or {}).items()
        }
        self._follower = ReferencePathFollower(
            goal_radius=self.goal_radius,
            turn_angle=self.turn_angle,
        )
        self._context: Optional[EpisodeContext] = None

    @staticmethod
    def _reference_path(episode: Any) -> List[List[float]]:
        path = []
        for waypoint in getattr(episode, "reference_path", None) or []:
            normalized = _position(waypoint)
            if normalized is not None:
                path.append(normalized)

        if not path:
            start = _position(getattr(episode, "start_position", None))
            if start is not None:
                path.append(start)
            goals = getattr(episode, "goals", None) or []
            if goals:
                goal = _position(getattr(goals[0], "position", goals[0]))
                if goal is not None:
                    path.append(goal)
        return path

    def reset(self, context: EpisodeContext) -> None:
        path = self._reference_path(context.episode)
        if len(path) < 2:
            raise ValueError(
                f"episode {context.episode_key!r} has no navigable reference path"
            )
        trajectory_type = getattr(context.episode, "trajectory_type", None)
        episode_goal_radius = self.success_distances.get(
            str(trajectory_type),
            self.success_distances.get("DEFAULT", self.goal_radius),
        )
        # ReferencePathFollower synchronizes this public property with its
        # nested waypoint follower.
        self._follower.goal_radius = episode_goal_radius
        self._context = context
        self._follower.reset(path)

    def act(self, observation: Mapping[str, Any]) -> PolicyStep:
        del observation
        if self._context is None:
            raise RuntimeError("reset() must be called before act()")
        action = self._follower.get_next_action(self._context.simulator)
        return PolicyStep(
            action=action,
            info={
                "policy": "reference_follower",
                "goal_radius": self._follower.goal_radius,
                "waypoint_index": self._follower.get_current_waypoint_index(),
                "progress": self._follower.get_progress(),
            },
        )

    def close(self) -> None:
        self._context = None
