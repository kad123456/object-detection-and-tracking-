import numpy as np

from src.tracker import TrackObservation
from src.utils import CrossingCounter, FPSCounter, TrailStore, draw_tracks, segments_intersect


def observation(track_id: int, center_x: float, center_y: float = 10.0) -> TrackObservation:
    return TrackObservation(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=0.9,
        x1=center_x - 3,
        y1=center_y - 3,
        x2=center_x + 3,
        y2=center_y + 3,
    )


def test_trail_is_bounded_and_draws_tracks() -> None:
    trails = TrailStore(max_length=2)
    trails.update([observation(1, 5)])
    trails.update([observation(1, 10)])
    trails.update([observation(1, 15)])
    assert trails.get(1) == [(10, 10), (15, 10)]

    frame = np.zeros((40, 40, 3), dtype=np.uint8)
    drawn = draw_tracks(frame, [observation(1, 15)], trails)
    assert drawn is frame
    assert np.count_nonzero(frame) > 0


def test_crossing_counter_counts_each_id_once() -> None:
    counter = CrossingCounter(((10, 0), (10, 20)))
    assert counter.update([observation(7, 5)]) == 0
    assert counter.update([observation(7, 15)]) == 1
    assert counter.update([observation(7, 5)]) == 1


def test_segments_intersect_finite_line_segment() -> None:
    assert segments_intersect((5, 5), (15, 5), (10, 0), (10, 10))
    assert not segments_intersect((5, 15), (15, 15), (10, 0), (10, 10))


def test_fps_counter_returns_average_rate() -> None:
    fps = FPSCounter()
    assert fps.tick(1.0) == 0.0
    assert fps.tick(2.0) == 2.0
