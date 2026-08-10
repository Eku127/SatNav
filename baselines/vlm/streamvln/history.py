"""Pure streaming-window and history selection used by the adapter."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple


def history_indices(
    current_step: int,
    *,
    num_history: Optional[int],
    future_stride: int,
) -> Tuple[int, ...]:
    """Return the pinned StreamVLN history sampling indices.

    This intentionally preserves upstream sampling: with a history cap the
    stride is ``max(current_step // num_history, 1)``; without a cap it is the
    action-chunk/future stride.
    """

    current_step = int(current_step)
    future_stride = int(future_stride)
    if current_step <= 0:
        return ()
    if future_stride <= 0:
        raise ValueError("future_stride must be positive")
    if num_history is None:
        stride = future_stride
    else:
        num_history = int(num_history)
        if num_history <= 0:
            raise ValueError("num_history must be positive or None")
        stride = max(current_step // num_history, 1)
    return tuple(range(0, current_step, stride))


@dataclass
class StreamingHistory:
    """Track observations and the active fixed-size StreamVLN window."""

    num_frames: int = 32
    frames: List[Any] = field(default_factory=list)
    window_time_ids: List[int] = field(default_factory=list)
    step_id: int = 0

    def __post_init__(self) -> None:
        self.num_frames = int(self.num_frames)
        if self.num_frames <= 0:
            raise ValueError("num_frames must be positive")

    def reset(self) -> None:
        self.frames.clear()
        self.window_time_ids.clear()
        self.step_id = 0

    def observe(self, frame: Any) -> None:
        """Record the pre-action observation for the current primitive step."""

        self.frames.append(frame)
        self.window_time_ids.append(self.step_id)

    def generation_frames(
        self,
        *,
        first_generation_in_window: bool,
        num_history: Optional[int],
        future_stride: int,
    ) -> List[Any]:
        """Return ``[history] + [current]`` with legacy boundary semantics.

        ``window_time_ids[0]`` is used deliberately.  If an action chunk crosses
        a 32-frame boundary, generation may happen a few primitive actions
        later; historical sampling must still be anchored at the boundary while
        the final element is the latest observation.
        """

        if not self.frames or not self.window_time_ids:
            raise RuntimeError("observe() must be called before generation")
        current = self.frames[-1]
        if not first_generation_in_window or self.step_id == 0:
            return [current]
        boundary_step = self.window_time_ids[0]
        selected = history_indices(
            boundary_step,
            num_history=num_history,
            future_stride=future_stride,
        )
        return [self.frames[index] for index in selected] + [current]

    def action_executed(self) -> bool:
        """Advance one primitive action and report a window boundary."""

        self.step_id += 1
        boundary = self.step_id % self.num_frames == 0
        if boundary:
            self.window_time_ids.clear()
        return boundary
