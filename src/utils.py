"""Drawing, frame-rate display, and line-crossing helpers."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from src.tracker import TrackObservation

Point = tuple[float, float]
Line = tuple[Point, Point]


def color_for_id(track_id: int) -> tuple[int, int, int]:
    """Return a stable, visible BGR color for a track ID."""
    value = (int(track_id) * 53 + 91) % 180
    return (80 + value // 2, 255 - value // 3, 80 + value // 2)


class TrailStore:
    """Keep a bounded history of center points for each persistent track ID."""

    def __init__(self, max_length: int = 24) -> None:
        if max_length < 1:
            raise ValueError("trail length must be at least 1")
        self._points: dict[int, deque[tuple[int, int]]] = {}
        self.max_length = max_length

    def update(self, tracks: Iterable[TrackObservation]) -> None:
        """Append one center point for each visible track."""
        for track in tracks:
            points = self._points.setdefault(track.track_id, deque(maxlen=self.max_length))
            points.append((round(track.center[0]), round(track.center[1])))

    def get(self, track_id: int) -> list[tuple[int, int]]:
        """Return a copy of the current center-point history."""
        return list(self._points.get(track_id, ()))


def draw_tracks(
    frame: np.ndarray,
    tracks: Iterable[TrackObservation],
    trails: TrailStore,
) -> np.ndarray:
    """Draw boxes, labels, and motion trails in place on a BGR frame."""
    height, width = frame.shape[:2]
    for track in tracks:
        color = color_for_id(track.track_id)
        x1 = max(0, min(width - 1, round(track.x1)))
        y1 = max(0, min(height - 1, round(track.y1)))
        x2 = max(0, min(width - 1, round(track.x2)))
        y2 = max(0, min(height - 1, round(track.y2)))
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{track.class_name} {track.confidence:.2f} ID:{track.track_id}"
        text_y = max(16, y1 - 7)
        cv2.putText(frame, label, (x1, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        points = trails.get(track.track_id)
        if len(points) > 1:
            cv2.polylines(frame, [np.asarray(points, dtype=np.int32)], False, color, 2)
    return frame


def draw_status(
    frame: np.ndarray,
    fps: float,
    crossing_count: int = 0,
    line: Line | None = None,
) -> np.ndarray:
    """Draw FPS and optional line-crossing status over a frame."""
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (12, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (30, 240, 30),
        2,
    )
    if line is not None:
        start = tuple(round(v) for v in line[0])
        end = tuple(round(v) for v in line[1])
        cv2.line(frame, start, end, (0, 220, 255), 2)
        cv2.putText(
            frame,
            f"Crossings: {crossing_count}",
            (12, 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 220, 255),
            2,
        )
    return frame


@dataclass
class FPSCounter:
    """Calculate an average frame rate over frames processed so far."""

    _frames: int = 0
    _started_at: float | None = None

    def tick(self, now: float) -> float:
        """Record a frame at monotonic time ``now`` and return average FPS."""
        if self._started_at is None:
            self._started_at = now
        self._frames += 1
        elapsed = now - self._started_at
        return self._frames / elapsed if elapsed > 0 else 0.0


def _orientation(a: Point, b: Point, c: Point) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a: Point, b: Point, p: Point) -> bool:
    return (
        min(a[0], b[0]) - 1e-9 <= p[0] <= max(a[0], b[0]) + 1e-9
        and min(a[1], b[1]) - 1e-9 <= p[1] <= max(a[1], b[1]) + 1e-9
    )


def segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool:
    """Return whether the two closed line segments intersect."""
    o1, o2 = _orientation(a, b, c), _orientation(a, b, d)
    o3, o4 = _orientation(c, d, a), _orientation(c, d, b)

    if o1 * o2 < 0 and o3 * o4 < 0:
        return True
    return (
        (abs(o1) <= 1e-9 and _on_segment(a, b, c))
        or (abs(o2) <= 1e-9 and _on_segment(a, b, d))
        or (abs(o3) <= 1e-9 and _on_segment(c, d, a))
        or (abs(o4) <= 1e-9 and _on_segment(c, d, b))
    )


class CrossingCounter:
    """Count each track ID once when its center crosses a configured segment."""

    def __init__(self, line: Line) -> None:
        self.line = line
        self._previous: dict[int, Point] = {}
        self._counted: set[int] = set()

    @property
    def count(self) -> int:
        """Number of unique track IDs that crossed the line."""
        return len(self._counted)

    def update(self, tracks: Iterable[TrackObservation]) -> int:
        """Update object centers and return the cumulative crossing count."""
        for track in tracks:
            current = track.center
            previous = self._previous.get(track.track_id)
            if (
                previous is not None
                and track.track_id not in self._counted
                and segments_intersect(previous, current, self.line[0], self.line[1])
            ):
                self._counted.add(track.track_id)
            self._previous[track.track_id] = current
        return self.count
